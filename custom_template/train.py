import torch
import torch.nn as nn
import time
import math
import random
import argparse
import collections
import numpy as np
from torchtext.data import Field, BucketIterator
from model import Transformer, Encoder, Decoder
from lora_model import lora_Transformer, lora_Encoder, lora_Decoder
from dataset import prepare_dataset
from utils import (
    epoch_time,
    count_parameters,
    insert_lora,
    initialize_weights,
    W_init_by_lora,
    W_init_by_SVD,
    make_W_zero,
    W_weight_copy,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import wandb
import loralib as lora
import copy
import os


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # current gpu seed
    torch.cuda.manual_seed_all(seed)  # All gpu seed
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # True로 하면 gpu에 적합한 알고리즘을 선택함.


seed_everything(42)


# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train()  # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스(<eos>)는 제외
        # 입력을 할 때는 <sos>부터 시작하도록 처리
        output, _ = model(src, trg[:, :-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0(<sos>)은 제외
        trg = trg[:, 1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward()  # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval()  # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output, _ = model(src, trg[:, :-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:, 1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def recon_error(original_weight, approx_weight):
    return torch.linalg.norm(original_weight - approx_weight, "fro")


def main(model_name, EX_type, rank):
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="lora-training",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": 0.0005,
    #         "epochs": 100,
    #     },
    # )
    # wandb.run.name = f"{model_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC, TRG, train_dataset, valid_dataset, test_dataset = prepare_dataset()

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HIDDEN_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    N_EPOCHS = 100
    CLIP = 1

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset), batch_size=BATCH_SIZE, device=device
    )
    enc = copy.deepcopy(Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device))
    dec = copy.deepcopy(Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device))
    lora_enc = lora_Encoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    lora_dec = lora_Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    transformer_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
    model = copy.deepcopy(transformer_model)

    if EX_type == "1":
        pass
    elif EX_type == "3":
        model.load_state_dict(
            torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
            strict=False,
        )

        insert_lora(model, HIDDEN_DIM, rank)

        print(f"The {model_name} has {count_parameters(model):,} trainable parameters")

        lora.mark_only_lora_as_trainable(model)

        W_model = copy.deepcopy(transformer_model)  # Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)
        W_model.load_state_dict(
            torch.load(
                "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer_copy.pt"
            ),
            strict=False,
        )
        W_weight_copy(model, W_model)

        print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
        print("### CHECK LAYERS WEIGHTS ###")
        print("W (encoder_q)\n", model.encoder.layers[0].self_attention.fc_q.weight.data)
        print(
            "encoder_q_LoraA\n",
            model.encoder.layers[0].self_attention.fc_q.lora_A,
            "encoder_q_LoraB\n",
            model.encoder.layers[0].self_attention.fc_q.lora_B,
        )

    elif EX_type == "4":
        model.load_state_dict(
            torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
            strict=False,
        )
        insert_lora(model, HIDDEN_DIM, rank)
        make_W_zero(model)
        lora.mark_only_lora_as_trainable(model)

        lora_model = copy.deepcopy(transformer_model)
        insert_lora(lora_model, HIDDEN_DIM, rank)
        lora_model.load_state_dict(
            torch.load(
                "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/EX3_LoRA_check_continuity_best_at17.pt"
            ),
            strict=False,
        )

        W_init_by_lora(model, lora_model)

        print(f"The {model_name} has {count_parameters(model):,} trainable parameters")

        print("### CHECK LAYERS WEIGHTS ###")
        print("W (encoder_q)\n", model.encoder.layers[0].self_attention.fc_q.weight.data)
        print(
            "encoder_q_LoraA\n",
            model.encoder.layers[0].self_attention.fc_q.lora_A,
            "encoder_q_LoraB\n",
            model.encoder.layers[0].self_attention.fc_q.lora_B,
        )
    elif EX_type == "5":
        model.load_state_dict(
            torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
            strict=False,
        )
        insert_lora(model, HIDDEN_DIM, rank)  # LoRA insert
        make_W_zero(model)  # W=0
        lora.mark_only_lora_as_trainable(model)  # F freeze

        SVD_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
        SVD_model.load_state_dict(
            torch.load(
                "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer_copy.pt"
            ),
            strict=False,
        )
        W_init_by_SVD(model, SVD_model, rank)  # dW init by W_lowrank

        print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
        print("### CHECK LAYERS WEIGHTS ###")
        print("W (encoder_q)\n", model.encoder.layers[0].self_attention.fc_q.weight.data)
        print(
            "encoder_q_LoraA\n",
            model.encoder.layers[0].self_attention.fc_q.lora_A,
            "encoder_q_LoraB\n",
            model.encoder.layers[0].self_attention.fc_q.lora_B,
        )
        print(
            "\n recon error: ",
            recon_error(
                SVD_model.encoder.layers[0].self_attention.fc_q.weight.data,
                model.encoder.layers[0].self_attention.fc_q.lora_A.transpose(0, 1)
                @ model.encoder.layers[0].self_attention.fc_q.lora_B.transpose(0, 1),
            ),
        )
    elif EX_type == "5_2":
        insert_lora(model, HIDDEN_DIM, rank)
        W_model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)
        W_model.load_state_dict(
            torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
            strict=False,
        )
        W_weight_copy(model, W_model)
        W_init_by_SVD(model, W_model, rank)
        make_W_zero(model)
        lora.mark_only_lora_as_trainable(model)

        print("### CHECK LAYERS WEIGHTS ###")
        print("W (encoder_q)\n", model.encoder.layers[0].self_attention.fc_q.weight.data)
        print(
            "encoder_q_LoraA\n",
            model.encoder.layers[0].self_attention.fc_q.lora_A,
            "encoder_q_LoraB\n",
            model.encoder.layers[0].self_attention.fc_q.lora_B,
        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 뒷 부분의 패딩(padding)에 대해서는 값 무시
    # criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=0.00001)

    # criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    # test_loss = evaluate(model, test_iterator, criterion)
    # print("### CHECK THE CONTINUITY ###")
    # print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
    # wandb.log({"Test Loss": test_loss, "Test PPL": math.exp(test_loss)})
    # best_valid_loss = float("inf")
    # not_improved_count = 0

    # for epoch in range(N_EPOCHS):
    #     start_time = time.time()  # 시작 시간 기록

    #     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    #     valid_loss = evaluate(model, valid_iterator, criterion)
    #     scheduler.step()
    #     end_time = time.time()  # 종료 시간 기록
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(
    #             model.state_dict(),
    #             f"/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/{model_name}_best_at{epoch}.pt",
    #         )
    #         not_improved_count = 0
    #     else:
    #         not_improved_count += 1

    #     print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
    #     print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}")
    #     print(f"\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}")
    #     wandb.log(
    #         {
    #             "Time": epoch_mins * 60 + epoch_secs,
    #             "Train loss": train_loss,
    #             "Train PPL": math.exp(train_loss),
    #             "Valid Loss": valid_loss,
    #             "Valid PPL": math.exp(valid_loss),
    #             "LR": scheduler.optimizer.param_groups[0]["lr"],  # scheduler.get_last_lr()[0]
    #         }
    #     )

    #     if not_improved_count == 5:
    #         print(f"Validation performance didn't improve for {not_improved_count} epochs at {epoch} epoch.")
    #         break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lora Training")
    parser.add_argument("-m", "--model", default="ex5Transformer", type=str, help="Model Name")
    parser.add_argument("-e", "--EX_type", default="5", type=str, help="Type of Experiment")
    parser.add_argument("-r", "--rank", default=64, type=int, help="Rank of LoRA")
    args = parser.parse_args()
    main(args.model, args.EX_type, args.rank)
