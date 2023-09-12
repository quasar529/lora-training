# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

import torch
import wandb
import loralib as lora


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recon_error(original_weight, approx_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.linalg.norm(original_weight.to(device) - approx_weight.to(device), "fro")


def insert_lora(model, dim, rank, lora_alpha):
    len_of_layers = 12  # len(model.roberta.encoder)
    for i in range(len_of_layers):
        model.roberta.encoder.layer[i].attention.self.query = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )
        model.roberta.encoder.layer[i].attention.self.value = copy.deepcopy(
            lora.Linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )


def W_weight_copy(new_model, W_model):
    len_of_layers = 12
    q_encoder_weight_list = []
    v_encoder_weight_list = []
    q_encoder_bias_list = []
    v_encoder_bias_list = []

    for i in range(len_of_layers):
        q_encoder_new_weight = W_model.roberta.encoder.layer[i].attention.self.query.weight.data
        q_encoder_weight_list.append(q_encoder_new_weight)
        q_encoder_new_bias = W_model.roberta.encoder.layer[i].attention.self.query.bias.data
        q_encoder_bias_list.append(q_encoder_new_bias)

        v_encoder_new_weight = W_model.roberta.encoder.layer[i].attention.self.value.weight.data
        v_encoder_weight_list.append(v_encoder_new_weight)
        v_encoder_new_bias = W_model.roberta.encoder.layer[i].attention.self.value.bias.data
        v_encoder_bias_list.append(v_encoder_new_bias)

    with torch.no_grad():
        for i in range(len_of_layers):
            new_model.roberta.encoder.layer[i].attention.self.query.weight.data.copy_(q_encoder_weight_list[i])
            new_model.roberta.encoder.layer[i].attention.self.value.weight.data.copy_(v_encoder_weight_list[i])
            new_model.roberta.encoder.layer[i].attention.self.query.bias.data.copy_(q_encoder_bias_list[i])
            new_model.roberta.encoder.layer[i].attention.self.value.bias.data.copy_(v_encoder_bias_list[i])


def make_W_zero(model):
    len_of_layers = 12  # len(model.encoder.layers)
    with torch.no_grad():
        for i in range(len_of_layers):
            model.roberta.encoder.layer[i].attention.self.query.weight.data.zero_()
            model.roberta.encoder.layer[i].attention.self.value.weight.data.zero_()


def dW_init_by_SVD(model, SVD_model, rank):
    w_q_encoder_loraA_weights = []
    w_q_encoder_loraB_weights = []

    w_v_encoder_loraA_weights = []
    w_v_encoder_loraB_weights = []

    len_of_layers = 12  # len(SVD_model.roberta.encoder.layer)
    with torch.no_grad():
        for i in range(len_of_layers):
            encoder_q_original_weight = SVD_model.roberta.encoder.layer[i].attention.self.query.weight.data
            encoder_v_original_weight = SVD_model.roberta.encoder.layer[i].attention.self.value.weight.data

            encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight)
            encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight)

            approx_rank = rank

            # w_q_encoder
            # torch.Size([768, rank])
            w_q_encoder_loraA_weights.append(
                encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt()
            )
            # torch.Size([rank, 768])
            w_q_encoder_loraB_weights.append(
                torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :]
            )
            # w_v_encoder
            w_v_encoder_loraA_weights.append(
                encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt()
            )
            w_v_encoder_loraB_weights.append(
                torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :]
            )
    og_weight = SVD_model.roberta.encoder.layer[0].attention.self.query.weight.data
    # insert_lora(SVD_model, 768, approx_rank, lora_alpha)

    with torch.no_grad():
        for i in range(len_of_layers):
            model.roberta.encoder.layer[i].attention.self.query.lora_A.copy_(
                w_q_encoder_loraA_weights[i].transpose(0, 1)
            )
            model.roberta.encoder.layer[i].attention.self.query.lora_B.copy_(
                w_q_encoder_loraB_weights[i].transpose(0, 1)
            )

            model.roberta.encoder.layer[i].attention.self.value.lora_A.copy_(
                w_v_encoder_loraA_weights[i].transpose(0, 1)
            )
            model.roberta.encoder.layer[i].attention.self.value.lora_B.copy_(
                w_v_encoder_loraB_weights[i].transpose(0, 1)
            )
    print(f"OG weight Norm : {torch.linalg.norm(og_weight)}")
    approx_weight = (
        model.roberta.encoder.layer[0].attention.self.query.lora_A.T
        @ model.roberta.encoder.layer[0].attention.self.query.lora_B.T
    )
    print(f"recon error between OG and rank_{approx_rank} SVD weight : {recon_error(og_weight,approx_weight)} ")


def dW_init_by_WplusAB(model, lora_model):
    len_of_layers = 12
    loraA_q_encoder_weight_list = []
    loraB_q_encoder_weight_list = []

    loraA_v_encoder_weight_list = []
    loraB_v_encoder_weight_list = []

    with torch.no_grad():
        for i in range(len_of_layers):
            loraA_q_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.query.lora_A
            loraA_q_encoder_weight_list.append(loraA_q_encoder_new_weight)
            loraB_q_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.query.lora_B
            loraB_q_encoder_weight_list.append(loraB_q_encoder_new_weight)

            loraA_v_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.value.lora_A
            loraA_v_encoder_weight_list.append(loraA_v_encoder_new_weight)
            loraB_v_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.value.lora_B
            loraB_v_encoder_weight_list.append(loraB_v_encoder_new_weight)

    with torch.no_grad():
        for i in range(len_of_layers):
            encoder_q_plus_AB = (
                lora_model.roberta.encoder.layer[i].attention.self.query.weight.data.T
                + (loraA_q_encoder_weight_list[i].T @ loraB_q_encoder_weight_list[i].T)
            ).T
            encoder_v_plus_AB = (
                lora_model.roberta.encoder.layer[i].attention.self.value.weight.data.T
                + (loraA_v_encoder_weight_list[i].T @ loraB_v_encoder_weight_list[i].T)
            ).T

            model.roberta.encoder.layer[i].attention.self.query.weight.copy_(encoder_q_plus_AB)

            model.roberta.encoder.layer[i].attention.self.value.weight.copy_(encoder_v_plus_AB)


def W_init_by_loraAB(model, lora_model):
    """
    model의 W weight를 lora_model의 Lora Layer의 weight로 초기화
    """
    len_of_layers = 12
    loraA_q_encoder_weight_list = []
    loraB_q_encoder_weight_list = []

    loraA_v_encoder_weight_list = []
    loraB_v_encoder_weight_list = []

    with torch.no_grad():
        for i in range(len_of_layers):
            loraA_q_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.query.lora_A
            loraA_q_encoder_weight_list.append(loraA_q_encoder_new_weight)
            loraB_q_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.query.lora_B
            loraB_q_encoder_weight_list.append(loraB_q_encoder_new_weight)

            loraA_v_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.value.lora_A
            loraA_v_encoder_weight_list.append(loraA_v_encoder_new_weight)
            loraB_v_encoder_new_weight = lora_model.roberta.encoder.layer[i].attention.self.value.lora_B
            loraB_v_encoder_weight_list.append(loraB_v_encoder_new_weight)

    with torch.no_grad():
        for i in range(len_of_layers):
            model.roberta.encoder.layer[i].attention.self.query.weight.copy_(
                (loraA_q_encoder_weight_list[i].T @ loraB_q_encoder_weight_list[i].T).T
            )

            model.roberta.encoder.layer[i].attention.self.value.weight.copy_(
                (loraA_v_encoder_weight_list[i].T @ loraB_v_encoder_weight_list[i].T).T
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--lora_r", type=int, default=None, help="Rank of LoRA")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    if args.do_train:
        wandb.init(
            group=f"dW init by SVD from HG ckpt",
            name=f"RANK{args.lora_r}_{args.model_name_or_path}_{args.task_name}",
            config={
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "batch_size": 64,
                "seed": 0,
                "lora_r": args.lora_r,
            },
        )
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")

    # Evaluation
    if not args.do_train and args.do_eval:
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")


if __name__ == "__main__":
    main()
