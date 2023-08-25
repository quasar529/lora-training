import torch
import torch.nn as nn
import time
import math
import random
import argparse
import collections
import numpy as np
from torchtext.data import Field, BucketIterator
from model import Transformer, Encoder, Decoder, TransformerClassifier
from lora_model import lora_Transformer, lora_Encoder, lora_Decoder
from dataset import prepare_dataset, IMDB, prepare_imdb
from utils import (
    epoch_time,
    count_parameters,
    insert_lora,
    insert_lora_encoderOnly,
    initialize_weights,
    W_init_by_lora,
    W_init_by_SVD,
    make_W_zero,
    make_W_zero_encoderOnly,
    W_weight_copy,
    W_weight_copy_encoderOnly,
    W_init_by_loraAB,
    W_init_by_WplusAB,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, StepLR
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


def binary_accuracy(preds, target):
    """
    from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    """
    # round predictions to the closest integer (0 or 1)
    rounded_preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)  # torch.round(torch.sigmoid(preds))

    # convert into float for division
    correct = (rounded_preds == target).float()

    # rounded_preds = [ 1   0   0   1   1   1   0   1   1   1]
    # targets       = [ 1   0   1   1   1   1   0   1   1   0]
    # correct       = [1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0]
    acc = correct.sum() / len(correct)
    return acc


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


def main(dataset_type, model_name, EX_type, rank):
    wandb.init(
        # set the wandb project where this run will be logged
        project="lora-training",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.0005,
            "epochs": 100,
        },
    )
    wandb.run.name = f"{model_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_type == "multi30k":
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
        N_EPOCHS = 200
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
            make_W_zero(model)
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
            wandb.log({"Trainable Parameters": count_parameters(model)})
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
            """
            W*를 SVD로 r = rank로 A,B로 분해한 후 W에 A@B를 넣어준다.
            단 이 때, 추후 진행할 5-1,5-2 실험에서는 lora layer를 사용하므로
            실험에 통일성을 지키기 위해
            W=0 으로 초기화 후 fix 하고,
            dW = A @ B, 즉 LoRA_A = A, LoRA_B = B로 초기화해 LoRA를 학습시킨다.
            """
            SVD_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
            SVD_model.load_state_dict(
                torch.load(
                    "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer_copy.pt"
                ),
                strict=False,
            )

            model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
                strict=False,
            )
            insert_lora(model, HIDDEN_DIM, rank)  # LoRA insert
            make_W_zero(model)  # W=0
            W_init_by_SVD(model, SVD_model, rank)  # dW init by W_lowrank
            lora.mark_only_lora_as_trainable(model)  # F freeze

            print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
            reconstruction_error = recon_error(
                SVD_model.encoder.layers[0].self_attention.fc_q.weight.data.T,
                model.encoder.layers[0].self_attention.fc_q.lora_A.transpose(0, 1)
                @ model.encoder.layers[0].self_attention.fc_q.lora_B.transpose(0, 1),
            )
            print("\n recon error: ", reconstruction_error)
            wandb.log(
                {"Trainable Parameters": count_parameters(model), "recon error of Encoder_fc_q": reconstruction_error}
            )
        elif EX_type == "5_1_2":
            ex5_2_1_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
            insert_lora(ex5_2_1_model, HIDDEN_DIM, rank)  # LoRA insert
            ex5_2_1_model.load_state_dict(
                torch.load(
                    "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/EX5_2_1_W=0_dW=A32B32_T_0=100.pt"
                ),
                strict=False,
            )
            ex5_2_1_model_criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
            test_loss = evaluate(ex5_2_1_model.to(device), test_iterator, ex5_2_1_model_criterion)
            print("### CHECK THE CONTINUITY ###")
            print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

            model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
                strict=False,
            )
            insert_lora(model, HIDDEN_DIM, 2)  # LoRA insert
            make_W_zero(model)  # W=0
            W_init_by_loraAB(model, ex5_2_1_model)
            lora.mark_only_lora_as_trainable(model)  # F freeze

            print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
            reconstruction_error = recon_error(
                model.encoder.layers[0].self_attention.fc_q.weight.data.T.to(device),
                ex5_2_1_model.encoder.layers[0].self_attention.fc_q.lora_A.transpose(0, 1)
                @ ex5_2_1_model.encoder.layers[0].self_attention.fc_q.lora_B.transpose(0, 1),
            )
            print("\n recon error: ", reconstruction_error)
            wandb.log(
                {"Trainable Parameters": count_parameters(model), "recon error of Encoder_fc_q": reconstruction_error}
            )
        elif EX_type == "5_1_2_2":
            ex5_1_2_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
            insert_lora(ex5_1_2_model, HIDDEN_DIM, 2)  # LoRA insert
            ex5_1_2_model.load_state_dict(
                torch.load(
                    "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/5_1_2_2_W=W+A2B2_dW=0_rank2_T_0=100_3rd_lr5e-4.pt"
                ),
                strict=False,
            )
            ex5_1_2_model_criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
            test_loss = evaluate(ex5_1_2_model.to(device), test_iterator, ex5_1_2_model_criterion)
            print("### CHECK THE CONTINUITY ###")
            print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

            model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
                strict=False,
            )
            insert_lora(model, HIDDEN_DIM, 2)  # LoRA insert
            make_W_zero(model)  # W=0
            W_init_by_WplusAB(model, ex5_1_2_model)
            lora.mark_only_lora_as_trainable(model)  # F freeze

            reconstruction_error = recon_error(
                model.encoder.layers[0].self_attention.fc_q.weight.data.T.to(device),
                ex5_1_2_model.encoder.layers[0].self_attention.fc_q.weight.data.T
                + (
                    ex5_1_2_model.encoder.layers[0].self_attention.fc_q.lora_A.transpose(0, 1)
                    @ ex5_1_2_model.encoder.layers[0].self_attention.fc_q.lora_B.transpose(0, 1)
                ),
            )
            print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
            print("\n recon error: ", reconstruction_error)
            wandb.log(
                {"Trainable Parameters": count_parameters(model), "recon error of Encoder_fc_q": reconstruction_error}
            )
        elif EX_type == "5_2_1":
            SVD_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
            SVD_model.load_state_dict(
                torch.load(
                    "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer_copy.pt"
                ),
                strict=False,
            )

            model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
                strict=False,
            )
            insert_lora(model, HIDDEN_DIM, rank)  # LoRA insert
            make_W_zero(model)  # W=0
            W_init_by_SVD(model, SVD_model, rank)  # dW init by W_lowrank
            lora.mark_only_lora_as_trainable(model)  # F freeze
            print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
            reconstruction_error = recon_error(
                SVD_model.encoder.layers[0].self_attention.fc_q.weight.data.T,
                model.encoder.layers[0].self_attention.fc_q.lora_A.transpose(0, 1)
                @ model.encoder.layers[0].self_attention.fc_q.lora_B.transpose(0, 1),
            )
            print("\n recon error of Encoder_fc_q: ", reconstruction_error)
            wandb.log(
                {"Trainable Parameters": count_parameters(model), "recon error of Encoder_fc_q": reconstruction_error}
            )

        elif EX_type == "5_2_2":
            ex5_2_1_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device))
            insert_lora(ex5_2_1_model, HIDDEN_DIM, rank)  # LoRA insert
            ex5_2_1_model.load_state_dict(
                torch.load(
                    "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/EX5_2_1_W=0_dW=A32B32_T_0=100.pt"
                ),
                strict=False,
            )
            ex5_2_1_model_criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
            test_loss = evaluate(ex5_2_1_model.to(device), test_iterator, ex5_2_1_model_criterion)
            print("### CHECK THE CONTINUITY ###")
            print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

            model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
                strict=False,
            )
            insert_lora(model, HIDDEN_DIM, rank)  # LoRA insert
            W_init_by_loraAB(model, ex5_2_1_model)
            lora.mark_only_lora_as_trainable(model)  # F freeze

            print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
            reconstruction_error = recon_error(
                model.encoder.layers[0].self_attention.fc_q.weight.data.T.to(device),
                ex5_2_1_model.encoder.layers[0].self_attention.fc_q.lora_A.transpose(0, 1)
                @ ex5_2_1_model.encoder.layers[0].self_attention.fc_q.lora_B.transpose(0, 1),
            )
            print("\n recon error: ", reconstruction_error)
            wandb.log(
                {"Trainable Parameters": count_parameters(model), "recon error of Encoder_fc_q": reconstruction_error}
            )

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0.00001)
        # 뒷 부분의 패딩(padding)에 대해서는 값 무시
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        test_loss = evaluate(model, test_iterator, criterion)
        print("### CHECK THE CONTINUITY ###")
        print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
        wandb.log({"Test Loss": test_loss, "Test PPL": math.exp(test_loss)})

        best_valid_loss = float("inf")
        not_improved_count = 0
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

        for epoch in range(N_EPOCHS):
            start_time = time.time()  # 시작 시간 기록

            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)
            scheduler.step()
            end_time = time.time()  # 종료 시간 기록
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(),
                    f"/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/{model_name}.pt",
                )
                wandb.log({"Best Epoch": epoch + 1})
                not_improved_count = 0
            else:
                not_improved_count += 1

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}")
            print(f"\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}")
            wandb.log(
                {
                    "Train loss": train_loss,
                    "Train PPL": math.exp(train_loss),
                    "Valid Loss": valid_loss,
                    "Valid PPL": math.exp(valid_loss),
                    "LR": scheduler.optimizer.param_groups[0]["lr"],  # scheduler.get_last_lr()[0]
                }
            )

            if not_improved_count == 10:
                print(f"Validation performance didn't improve for {not_improved_count} epochs at {epoch+1} epoch.")
                criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
                test_loss = evaluate(model, test_iterator, criterion)
                print("### AFTER TRAIN, TEST LOSS###")
                print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
                wandb.log({"Test Loss": test_loss, "Test PPL": math.exp(test_loss)})
                break

    elif dataset_type == "imdb":
        TEXT, LABEL, train_iter, test_iter = prepare_imdb()
        INPUT_DIM = len(TEXT.vocab)
        OUTPUT_DIM = 2
        HIDDEN_DIM = 128
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        SRC_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        MAX_LENGTH = 100
        enc = copy.deepcopy(
            Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, MAX_LENGTH)
        )
        model = copy.deepcopy(TransformerClassifier(enc, HIDDEN_DIM, OUTPUT_DIM, SRC_PAD_IDX, ENC_DROPOUT, device))

        LR = 1e-4
        EPOCHS = 200

        best_test_loss = float("inf")
        if EX_type == "1":
            model.apply(initialize_weights)
            print(f"The model has {count_parameters(model):,} trainable parameters")
            wandb.log({"Trainable Parameters": count_parameters(model)})
        elif EX_type == "3":
            model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/EX1IMDB.pt"),
                strict=False,
            )

            insert_lora_encoderOnly(model, HIDDEN_DIM, rank)
            make_W_zero_encoderOnly(model)
            lora.mark_only_lora_as_trainable(model)

            W_model = copy.deepcopy(
                TransformerClassifier(enc, HIDDEN_DIM, OUTPUT_DIM, SRC_PAD_IDX, ENC_DROPOUT, device)
            )  # Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)
            W_model.load_state_dict(
                torch.load("/content/drive/MyDrive/LAB/EX1IMDB_copy.pt"),
                strict=False,
            )
            W_weight_copy_encoderOnly(model, W_model)

            print(f"The {model_name} has {count_parameters(model):,} trainable parameters")
            print("### CHECK LAYERS WEIGHTS ###")
            print("W (encoder_q)\n", model.encoder.layers[0].self_attention.fc_q.weight.data)
            print(
                "encoder_q_LoraA\n",
                model.encoder.layers[0].self_attention.fc_q.lora_A,
                "encoder_q_LoraB\n",
                model.encoder.layers[0].self_attention.fc_q.lora_B,
            )
            wandb.log({"Trainable Parameters": count_parameters(model)})
        elif EX_type == "5":
            pass
        elif EX_type == "5-1":
            pass
        elif EX_type == "5-1":
            pass
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        model.to(device)

        for epoch in range(EPOCHS):
            start_time = time.time()  # 시작 시간 기록
            epoch_loss = 0
            epoch_acc = 0

            epoch_correct = 0
            epoch_count = 0

            not_improved_cnt = 0
            best_test_loss = float("inf")
            model.train()
            ### TRAIN
            for idx, batch in enumerate(iter(train_iter)):
                predictions = model(batch.text.to(device))
                labels = batch.label.to(device) - 1

                loss = criterion(predictions, labels)
                acc = binary_accuracy(predictions, labels)
                correct = predictions.argmax(axis=1) == labels

                epoch_correct += correct.sum().item()
                epoch_count += correct.size(0)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
            ### TEST
            with torch.no_grad():
                test_epoch_loss = 0
                test_acc = 0
                test_epoch_correct = 0
                test_epoch_count = 0
                model.eval()
                for idx, batch in enumerate(iter(test_iter)):
                    predictions = model(batch.text.to(device))
                    labels = batch.label.to(device) - 1
                    test_loss = criterion(predictions, labels)
                    acc = binary_accuracy(predictions, labels)

                    correct = predictions.argmax(axis=1) == labels

                    test_epoch_correct += correct.sum().item()
                    test_epoch_count += correct.size(0)
                    test_epoch_loss += loss.item()
                    test_acc += acc.item()
            end_time = time.time()  # 종료 시간 기록
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")

            print(f"epoch_loss = {epoch_loss:5f}")
            print(f"epoch accuracy = {(epoch_acc / len(train_iter)):5f}")
            print(f"test_epoch_loss = {test_epoch_loss:5f}")
            print(f"test epoch accuracy = {(test_acc / len(test_iter)):5f}")
            if (test_acc / len(test_iter)) < best_test_loss:
                best_test_loss = test_epoch_correct / test_epoch_count
                torch.save(model.state_dict(), f"/content/drive/MyDrive/LAB/{model_name}.pt")
                not_improved_cnt = 0
                wandb.log({"Best Epoch": epoch + 1})
            else:
                not_improved_cnt += 1

            wandb.log(
                {
                    "Train LOSS": epoch_loss,
                    "Train ACC": epoch_acc / len(train_iter),
                    "TEST LOSS": test_epoch_loss,
                    "TEST ACC": test_acc / len(test_iter),
                    "LR": scheduler.optimizer.param_groups[0]["lr"],  # scheduler.get_last_lr()[0]
                }
            )

            if not_improved_cnt == 10:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lora Training")
    parser.add_argument("-d", "--dataset_type", default="imdb", type=str, help="what kind of dataset to use")
    parser.add_argument(
        "-m", "--model", default="EX3_IMDB_TransformerClassifier", type=str, help="Model Name"
    )  # 5_1_2_2_W=W+A2B2_dW=0_rank2_T_0=100_3rd_lr5e-4
    parser.add_argument("-e", "--EX_type", default="3", type=str, help="Type of Experiment")
    parser.add_argument("-r", "--rank", default=32, type=int, help="Rank of LoRA")
    args = parser.parse_args()
    main(args.dataset_type, args.model, args.EX_type, args.rank)

### util 함수 구현 전 ###
# with torch.no_grad():
#     for i in range(3):
#         encoder_q_original_weight = model.encoder.layers[i].self_attention.fc_q.weight.data
#         encoder_v_original_weight = model.encoder.layers[i].self_attention.fc_v.weight.data

#         decoder_q_original_weight = model.decoder.layers[i].self_attention.fc_q.weight.data
#         decoder_v_original_weight = model.decoder.layers[i].self_attention.fc_v.weight.data

#         encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight)
#         encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight)

#         decoder_q_u, decoder_q_s, decoder_q_v = torch.linalg.svd(decoder_q_original_weight)
#         decoder_v_u, decoder_v_s, decoder_v_v = torch.linalg.svd(decoder_v_original_weight)
# make_W_zero(model)
# with torch.no_grad():
#     for i in range(3):
#         approx_rank = 32
#         # W = low rank W
#         model.encoder.layers[i].self_attention.fc_q.weight.copy_(
#             (encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt())
#             @ (torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :])
#         )  # = encoder_q_u @ torch.diag(encoder_q_s) @ encoder_q_v
#         model.encoder.layers[i].self_attention.fc_v.weight.copy_(
#             (encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt())
#             @ (torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :])
#         )  # .data = encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v
#         model.decoder.layers[i].self_attention.fc_q.weight.copy_(
#             (decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt())
#             @ (torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :])
#         )
#         model.decoder.layers[i].self_attention.fc_v.weight.copy_(
#             (decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt())
#             @ (torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :])
#         )

# insert_lora(model, HIDDEN_DIM, 32)
# lora.mark_only_lora_as_trainable(model)

# print("### CHECK LAYERS WEIGHTS ###")
# print("W (encoder_q)\n", model.encoder.layers[0].self_attention.fc_q.weight.data)
# print(
#     "encoder_q_LoraA\n",
#     model.encoder.layers[0].self_attention.fc_q.lora_A,
#     "encoder_q_LoraB\n",
#     model.encoder.layers[0].self_attention.fc_q.lora_B,
# )
