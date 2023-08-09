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
    recon_error,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import wandb
import loralib as lora
from train import evaluate
import os
import copy


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


def recon_error(original_weight, approx_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.linalg.norm(original_weight.to(device) - approx_weight.to(device), "fro")


def main():
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
    ENC_DROPOUT = 0  # 0.1
    DEC_DROPOUT = 0  # 0.1
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    N_EPOCHS = 100
    CLIP = 1
    # INPUT_DIM = len(SRC.vocab)
    # OUTPUT_DIM = len(TRG.vocab)
    # HIDDEN_DIM = 4
    # ENC_LAYERS = 1
    # DEC_LAYERS = 1
    # ENC_HEADS = 1
    # DEC_HEADS = 1
    # ENC_PF_DIM = 1
    # DEC_PF_DIM = 1
    # ENC_DROPOUT = 0  # 0.1
    # DEC_DROPOUT = 0  # 0.1
    # BATCH_SIZE = 2
    # LEARNING_RATE = 0.005
    # N_EPOCHS = 100
    # CLIP = 1

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset), batch_size=BATCH_SIZE, device=device
    )
    enc = copy.deepcopy(Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device))
    dec = copy.deepcopy(Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device))

    # rank_list = [32, 64, 128, 256]
    # s_of_svd = []
    # for rank in rank_list:
    #     print(f"{rank} Rank Model ")
    #     model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device))
    #     w_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device))
    #     model.load_state_dict(
    #         torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
    #         strict=False,
    #     )
    #     w_model.load_state_dict(
    #         torch.load(
    #             "/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer_copy.pt"
    #         ),
    #         strict=False,
    #     )
    #     insert_lora(model, HIDDEN_DIM, rank)
    #     make_W_zero(model)

    #     with torch.no_grad():
    #         for i in range(3):
    #             encoder_q_original_weight = w_model.encoder.layers[i].self_attention.fc_q.weight.data
    #             encoder_v_original_weight = w_model.encoder.layers[i].self_attention.fc_v.weight.data

    #             decoder_q_original_weight = w_model.decoder.layers[i].self_attention.fc_q.weight.data
    #             decoder_v_original_weight = w_model.decoder.layers[i].self_attention.fc_v.weight.data

    #             encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight.T)
    #             encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight.T)

    #             decoder_q_u, decoder_q_s, decoder_q_v = torch.linalg.svd(decoder_q_original_weight.T)
    #             decoder_v_u, decoder_v_s, decoder_v_v = torch.linalg.svd(decoder_v_original_weight.T)

    #             approx_rank = rank

    #             model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(
    #                 (encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt()).T
    #             )
    #             model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(
    #                 (torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :]).T
    #             )
    #             model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(
    #                 (encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt()).T
    #             )
    #             model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(
    #                 (torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :]).T
    #             )

    #             model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(
    #                 (decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt()).T
    #             )
    #             model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(
    #                 (torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :]).T
    #             )

    #             model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(
    #                 (decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt()).T
    #             )
    #             model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(
    #                 (torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :]).T
    #             )
    #             print("Original Weight Norm: ", torch.linalg.norm(encoder_q_original_weight.T))
    #             print(
    #                 "Reconstruction Error : Original Encoder FC_Q - FC_Q.loraA.T @ FC_Q.loraB.T",
    #                 recon_error(
    #                     encoder_q_original_weight.T,
    #                     model.encoder.layers[i].self_attention.fc_q.lora_A.T
    #                     @ model.encoder.layers[i].self_attention.fc_q.lora_B.T,
    #                 ),
    #             )
    #     criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    #     model.to(device)
    #     test_loss = evaluate(model, test_iterator, criterion)

    #     print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}\n")

    ### 여기부터
    model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device))
    w_model = copy.deepcopy(Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device))
    model.load_state_dict(
        torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt"),
        strict=False,
    )
    w_model.load_state_dict(
        torch.load("/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer_copy.pt"),
        strict=False,
    )

    rank = 64
    insert_lora(model, HIDDEN_DIM, rank)

    make_W_zero(model)
    # print(model.encoder.layers[0].self_attention.fc_q.weight.data)
    W_init_by_SVD(model, w_model, rank)

    # with torch.no_grad():
    #     for i in range(3):
    #         encoder_q_original_weight = w_model.encoder.layers[i].self_attention.fc_q.weight.data
    #         encoder_v_original_weight = w_model.encoder.layers[i].self_attention.fc_v.weight.data

    #         decoder_q_original_weight = w_model.decoder.layers[i].self_attention.fc_q.weight.data
    #         decoder_v_original_weight = w_model.decoder.layers[i].self_attention.fc_v.weight.data

    #         encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight.T)
    #         encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight.T)

    #         decoder_q_u, decoder_q_s, decoder_q_v = torch.linalg.svd(decoder_q_original_weight.T)
    #         decoder_v_u, decoder_v_s, decoder_v_v = torch.linalg.svd(decoder_v_original_weight.T)

    #         approx_rank = rank

    #         model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(
    #             (encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt()).T
    #         )
    #         model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(
    #             (torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :]).T
    #         )
    #         model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(
    #             (encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt()).T
    #         )
    #         model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(
    #             (torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :]).T
    #         )

    #         model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(
    #             (decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt()).T
    #         )
    #         model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(
    #             (torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :]).T
    #         )

    #         model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(
    #             (decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt()).T
    #         )
    #         model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(
    #             (torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :]).T
    #         )

    #         print(
    #             recon_error(
    #                 encoder_q_original_weight,
    #                 model.encoder.layers[i].self_attention.fc_q.lora_A.T
    #                 @ model.encoder.layers[i].self_attention.fc_q.lora_B.T,
    #             )
    #         )

    ### 여기까지

    # model.encoder.layers[i].self_attention.fc_q.weight.copy_((encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt())@(torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :])) #= encoder_q_u @ torch.diag(encoder_q_s) @ encoder_q_v
    # model.encoder.layers[i].self_attention.fc_v.weight.copy_((encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt())@(torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :]))#.data = encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v
    # model.decoder.layers[i].self_attention.fc_q.weight.copy_((decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt())@(torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :]))
    # model.decoder.layers[i].self_attention.fc_v.weight.copy_((decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt())@(torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :]))

    # W = 0

    # with torch.no_grad():
    #     for i in range(3):
    #         approx_rank = 64

    #         # model.encoder.layers[i].self_attention.fc_q.weight.copy_(encoder_q_u @ torch.diag(encoder_q_s).sqrt()) @ encoder_q_v) #= encoder_q_u @ torch.diag(encoder_q_s) @ encoder_q_v
    #         # model.encoder.layers[i].self_attention.fc_v.weight.copy_(encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v)#.data = encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v
    #         # model.decoder.layers[i].self_attention.fc_q.weight.copy_(decoder_q_u @ torch.diag(decoder_q_s) @ decoder_q_v)#.data = decoder_q_u @ torch.diag(decoder_q_s) @ decoder_q_v
    #         # model.decoder.layers[i].self_attention.fc_v.weight.copy_(decoder_v_u @ torch.diag(decoder_v_s) @ decoder_v_v)#.data = decoder_v_u @ torch.diag(decoder_v_s) @ decoder_v_v

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
    # W svd
    # with torch.no_grad():
    #     for i in range(ENC_LAYERS):
    #         encoder_q_original_weight = w_model.encoder.layers[i].self_attention.fc_q.weight.data
    #         encoder_v_original_weight = w_model.encoder.layers[i].self_attention.fc_v.weight.data

    #         decoder_q_original_weight = w_model.decoder.layers[i].self_attention.fc_q.weight.data
    #         decoder_v_original_weight = w_model.decoder.layers[i].self_attention.fc_v.weight.data

    #         encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight)
    #         encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight)

    #         decoder_q_u, decoder_q_s, decoder_q_v = torch.linalg.svd(decoder_q_original_weight)
    #         decoder_v_u, decoder_v_s, decoder_v_v = torch.linalg.svd(decoder_v_original_weight)

    #         approx_rank = rank

    #         model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(
    #             (encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt()).T
    #         )
    #         model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(
    #             (torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :]).T
    #         )
    #         model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(
    #             (encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt()).T
    #         )
    #         model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(
    #             (torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :]).T
    #         )

    #         model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(
    #             (decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt()).T
    #         )
    #         model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(
    #             (torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :]).T
    #         )
    #         model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(
    #             (decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt()).T
    #         )
    #         model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(
    #             (torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :]).T
    #         )
    #         # model.encoder.layers[i].self_attention.fc_q.weight.copy_(encoder_q_u @ torch.diag(encoder_q_s).sqrt()) @ encoder_q_v) #= encoder_q_u @ torch.diag(encoder_q_s) @ encoder_q_v
    #         # model.encoder.layers[i].self_attention.fc_v.weight.copy_(encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v)#.data = encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v
    #         # model.decoder.layers[i].self_attention.fc_q.weight.copy_(decoder_q_u @ torch.diag(decoder_q_s) @ decoder_q_v)#.data = decoder_q_u @ torch.diag(decoder_q_s) @ decoder_q_v
    #         # model.decoder.layers[i].self_attention.fc_v.weight.copy_(decoder_v_u @ torch.diag(decoder_v_s) @ decoder_v_v)#.data = decoder_v_u @ torch.diag(decoder_v_s) @ decoder_v_v

    #         w_model.encoder.layers[i].self_attention.fc_q.weight.copy_(
    #             (encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]) @ encoder_q_v[:approx_rank, :]).T
    #         )
    #         print(
    #             "SVD Valid Test: ",
    #             recon_error(
    #                 w_model.encoder.layers[i].self_attention.fc_q.weight,
    #                 model.encoder.layers[i].self_attention.fc_q.lora_A.T
    #                 @ model.encoder.layers[i].self_attention.fc_q.lora_B.T,
    #             ),
    #         )

    #         w_model.encoder.layers[i].self_attention.fc_v.weight.copy_(
    #             (
    #                 encoder_v_u[:, :approx_rank]
    #                 @ (torch.diag(encoder_v_s[:approx_rank]) @ encoder_v_v[:approx_rank, :])
    #             ).T
    #         )
    #         print(
    #             "SVD Valid Test: ",
    #             recon_error(
    #                 w_model.encoder.layers[i].self_attention.fc_v.weight,
    #                 model.encoder.layers[i].self_attention.fc_v.lora_A.T
    #                 @ model.encoder.layers[i].self_attention.fc_v.lora_B.T,
    #             ),
    #         )

    #         w_model.decoder.layers[i].self_attention.fc_q.weight.copy_(
    #             (
    #                 decoder_q_u[:, :approx_rank]
    #                 @ (torch.diag(decoder_q_s[:approx_rank]))
    #                 @ decoder_q_v[:approx_rank, :]
    #             ).T
    #         )
    #         print(
    #             "SVD Valid Test: ",
    #             recon_error(
    #                 w_model.decoder.layers[i].self_attention.fc_q.weight,
    #                 model.decoder.layers[i].self_attention.fc_q.lora_A.T
    #                 @ model.decoder.layers[i].self_attention.fc_q.lora_B.T,
    #             ),
    #         )

    #         w_model.decoder.layers[i].self_attention.fc_v.weight.copy_(
    #             (
    #                 decoder_v_u[:, :approx_rank]
    #                 @ (torch.diag(decoder_v_s[:approx_rank]))
    #                 @ decoder_v_v[:approx_rank, :]
    #             ).T
    #         )
    #         print(
    #             "SVD Valid Test: ",
    #             recon_error(
    #                 w_model.decoder.layers[i].self_attention.fc_v.weight,
    #                 model.decoder.layers[i].self_attention.fc_v.lora_A.T
    #                 @ model.decoder.layers[i].self_attention.fc_v.lora_B.T,
    #             ),
    #         )

    # with torch.no_grad():
    #     for i in range(1):
    #         encoder_q_original_weight = model.encoder.layers[i].self_attention.fc_q.weight.data
    #         encoder_v_original_weight = model.encoder.layers[i].self_attention.fc_v.weight.data

    #         decoder_q_original_weight = model.decoder.layers[i].self_attention.fc_q.weight.data
    #         decoder_v_original_weight = model.decoder.layers[i].self_attention.fc_v.weight.data

    #         encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight)
    #         encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight)

    #         decoder_q_u, decoder_q_s, decoder_q_v = torch.linalg.svd(decoder_q_original_weight)
    #         decoder_v_u, decoder_v_s, decoder_v_v = torch.linalg.svd(decoder_v_original_weight)

    #         approx_rank = 2

    # model.encoder.layers[i].self_attention.fc_q.weight.copy_(encoder_q_u @ torch.diag(encoder_q_s).sqrt()) @ encoder_q_v) #= encoder_q_u @ torch.diag(encoder_q_s) @ encoder_q_v
    # model.encoder.layers[i].self_attention.fc_v.weight.copy_(encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v)#.data = encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v
    # model.decoder.layers[i].self_attention.fc_q.weight.copy_(decoder_q_u @ torch.diag(decoder_q_s) @ decoder_q_v)#.data = decoder_q_u @ torch.diag(decoder_q_s) @ decoder_q_v
    # model.decoder.layers[i].self_attention.fc_v.weight.copy_(decoder_v_u @ torch.diag(decoder_v_s) @ decoder_v_v)#.data = decoder_v_u @ torch.diag(decoder_v_s) @ decoder_v_v

    # make_W_zero(model)
    # with torch.no_grad():
    #     for i in range(1):
    #         approx_rank = 2
    #         # W = low rank W
    #         # model.encoder.layers[i].self_attention.fc_q.weight.copy_(
    #         #     (encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt())
    #         #     @ (torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :])
    #         # )  # = encoder_q_u @ torch.diag(encoder_q_s) @ encoder_q_v
    #         # model.encoder.layers[i].self_attention.fc_v.weight.copy_(
    #         #     (encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt())
    #         #     @ (torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :])
    #         # )  # .data = encoder_v_u @ torch.diag(encoder_v_s) @ encoder_v_v
    #         # model.decoder.layers[i].self_attention.fc_q.weight.copy_(
    #         #     (decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt())
    #         #     @ (torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :])
    #         # )
    #         # model.decoder.layers[i].self_attention.fc_v.weight.copy_(
    #         #     (decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt())
    #         #     @ (torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :])
    #         # )
    #         base_model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(w_q_encoder_loraA_weights[i].transpose(0, 1))
    #         base_model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(w_q_encoder_loraB_weights[i].transpose(0, 1))

    #         base_model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(w_v_encoder_loraA_weights[i].transpose(0, 1))
    #         base_model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(w_v_encoder_loraB_weights[i].transpose(0, 1))

    #         base_model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(w_q_decoder_loraA_weights[i].transpose(0, 1))
    #         base_model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(w_q_decoder_loraB_weights[i].transpose(0, 1))

    #         base_model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(w_v_decoder_loraA_weights[i].transpose(0, 1))
    #         base_model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(w_v_decoder_loraB_weights[i].transpose(0, 1))
    # W_init_by_SVD(model, w_model, 2)

    # w_model.eval()
    # w_criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    # w_model.to(device)
    # w_test_loss = evaluate(w_model, test_iterator, w_criterion)
    # print("w_model")
    # print(f"Test Loss: {w_test_loss:.3f} | Test PPL: {math.exp(w_test_loss):.3f}")

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    model.to(device)
    test_loss = evaluate(model, test_iterator, criterion)
    print("model")
    print(f"Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")


if __name__ == "__main__":
    main()
