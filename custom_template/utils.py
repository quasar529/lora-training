import torch.nn as nn
import math
import time
import torch
import loralib as lora
import torch.nn.functional as F
from lora_layers import Linear as lora_linear
import copy


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def insert_lora(model, dim, rank, lora_alpha=1):
    len_of_layers = len(model.encoder.layers)
    for i in range(len_of_layers):
        model.encoder.layers[i].self_attention.fc_q = copy.deepcopy(
            lora_linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )
        model.encoder.layers[i].self_attention.fc_v = copy.deepcopy(
            lora_linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )

        model.decoder.layers[i].self_attention.fc_q = copy.deepcopy(
            lora_linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )
        model.decoder.layers[i].self_attention.fc_v = copy.deepcopy(
            lora_linear(dim, dim, r=rank, lora_alpha=lora_alpha, merge_weights=False)
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_W_zero(model):
    len_of_layers = len(model.encoder.layers)
    with torch.no_grad():
        for i in range(len_of_layers):
            model.encoder.layers[i].self_attention.fc_q.weight.data.zero_()
            model.encoder.layers[i].self_attention.fc_v.weight.data.zero_()
            model.decoder.layers[i].self_attention.fc_q.weight.data.zero_()
            model.decoder.layers[i].self_attention.fc_v.weight.data.zero_()


def W_weight_copy(new_model, W_model):
    len_of_layers = len(new_model.encoder.layers)
    q_encoder_weight_list = []
    v_encoder_weight_list = []

    q_decoder_weight_list = []
    v_decoder_weight_list = []

    for i in range(len_of_layers):
        q_encoder_new_weight = W_model.encoder.layers[i].self_attention.fc_q.weight.data
        q_encoder_weight_list.append(q_encoder_new_weight)
        v_encoder_new_weight = W_model.encoder.layers[i].self_attention.fc_v.weight.data
        v_encoder_weight_list.append(v_encoder_new_weight)

        q_decoder_new_weight = W_model.decoder.layers[i].self_attention.fc_q.weight.data
        q_decoder_weight_list.append(q_decoder_new_weight)
        v_decoder_new_weight = W_model.decoder.layers[i].self_attention.fc_v.weight.data
        v_decoder_weight_list.append(v_decoder_new_weight)
    with torch.no_grad():
        for i in range(len_of_layers):
            new_model.encoder.layers[i].self_attention.fc_q.weight.data.copy_(q_encoder_weight_list[i])
            new_model.encoder.layers[i].self_attention.fc_v.weight.data.copy_(v_encoder_weight_list[i])

            new_model.decoder.layers[i].self_attention.fc_q.weight.data.copy_(q_decoder_weight_list[i])
            new_model.decoder.layers[i].self_attention.fc_v.weight.data.copy_(v_decoder_weight_list[i])


def W_init_by_lora(base_model, lora_model):
    """
    base_model의 W weight를 lora_model의 Lora Layer의 weight로 초기화
    """
    len_of_layers = len(base_model.encoder.layers)
    loraA_q_encoder_weight_list = []
    loraB_q_encoder_weight_list = []

    loraA_v_encoder_weight_list = []
    loraB_v_encoder_weight_list = []

    loraA_q_decoder_weight_list = []
    loraB_q_decoder_weight_list = []

    loraA_v_decoder_weight_list = []
    loraB_v_decoder_weight_list = []
    with torch.no_grad():
        for i in range(len_of_layers):
            loraA_q_encoder_new_weight = lora_model.encoder.layers[i].self_attention.fc_q.lora_A
            loraA_q_encoder_weight_list.append(loraA_q_encoder_new_weight)
            loraB_q_encoder_new_weight = lora_model.encoder.layers[i].self_attention.fc_q.lora_B
            loraB_q_encoder_weight_list.append(loraB_q_encoder_new_weight)

            loraA_v_encoder_new_weight = lora_model.encoder.layers[i].self_attention.fc_v.lora_A
            loraA_v_encoder_weight_list.append(loraA_v_encoder_new_weight)
            loraB_v_encoder_new_weight = lora_model.encoder.layers[i].self_attention.fc_v.lora_B
            loraB_v_encoder_weight_list.append(loraB_v_encoder_new_weight)

            loraA_q_decoder_new_weight = lora_model.decoder.layers[i].self_attention.fc_q.lora_A
            loraA_q_decoder_weight_list.append(loraA_q_decoder_new_weight)
            loraB_q_decoder_new_weight = lora_model.decoder.layers[i].self_attention.fc_q.lora_B
            loraB_q_decoder_weight_list.append(loraB_q_decoder_new_weight)

            loraA_v_decoder_new_weight = lora_model.decoder.layers[i].self_attention.fc_v.lora_A
            loraA_v_decoder_weight_list.append(loraA_v_decoder_new_weight)
            loraB_v_decoder_new_weight = lora_model.decoder.layers[i].self_attention.fc_v.lora_B
            loraB_v_decoder_weight_list.append(loraB_v_decoder_new_weight)

    with torch.no_grad():
        for i in range(len_of_layers):
            base_model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(loraA_q_encoder_weight_list[i])
            base_model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(loraB_q_encoder_weight_list[i])
            base_model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(loraA_v_encoder_weight_list[i])
            base_model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(loraB_v_encoder_weight_list[i])

            base_model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(loraA_q_decoder_weight_list[i])
            base_model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(loraB_q_decoder_weight_list[i])
            base_model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(loraA_v_decoder_weight_list[i])
            base_model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(loraB_v_decoder_weight_list[i])


def recon_error(original_weight, approx_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.linalg.norm(original_weight.to(device) - approx_weight.to(device), "fro")


def W_init_by_SVD(base_model, SVD_model, rank):
    w_q_encoder_loraA_weights = []
    w_q_encoder_loraB_weights = []

    w_v_encoder_loraA_weights = []
    w_v_encoder_loraB_weights = []

    w_q_decoder_loraA_weights = []
    w_q_decoder_loraB_weights = []

    w_v_decoder_loraA_weights = []
    w_v_decoder_loraB_weights = []
    len_of_layers = len(base_model.encoder.layers)
    with torch.no_grad():
        for i in range(len_of_layers):
            encoder_q_original_weight = SVD_model.encoder.layers[i].self_attention.fc_q.weight.data
            encoder_v_original_weight = SVD_model.encoder.layers[i].self_attention.fc_v.weight.data

            decoder_q_original_weight = SVD_model.decoder.layers[i].self_attention.fc_q.weight.data
            decoder_v_original_weight = SVD_model.decoder.layers[i].self_attention.fc_v.weight.data

            encoder_q_u, encoder_q_s, encoder_q_v = torch.linalg.svd(encoder_q_original_weight)
            encoder_v_u, encoder_v_s, encoder_v_v = torch.linalg.svd(encoder_v_original_weight)

            decoder_q_u, decoder_q_s, decoder_q_v = torch.linalg.svd(decoder_q_original_weight)
            decoder_v_u, decoder_v_s, decoder_v_v = torch.linalg.svd(decoder_v_original_weight)

            approx_rank = rank
            # w_q_encoder
            w_q_encoder_loraA_weights.append(
                encoder_q_u[:, :approx_rank] @ torch.diag(encoder_q_s[:approx_rank]).sqrt()
            )  # torch.Size([256, 64])
            w_q_encoder_loraB_weights.append(
                torch.diag(encoder_q_s[:approx_rank]).sqrt() @ encoder_q_v[:approx_rank, :]
            )  # torch.Size([64, 256])
            # w_v_encoder
            w_v_encoder_loraA_weights.append(
                encoder_v_u[:, :approx_rank] @ torch.diag(encoder_v_s[:approx_rank]).sqrt()
            )
            w_v_encoder_loraB_weights.append(
                torch.diag(encoder_v_s[:approx_rank]).sqrt() @ encoder_v_v[:approx_rank, :]
            )
            # w_q_decoder
            w_q_decoder_loraA_weights.append(
                decoder_q_u[:, :approx_rank] @ torch.diag(decoder_q_s[:approx_rank]).sqrt()
            )
            w_q_decoder_loraB_weights.append(
                torch.diag(decoder_q_s[:approx_rank]).sqrt() @ decoder_q_v[:approx_rank, :]
            )
            # w_v_decoder
            w_v_decoder_loraA_weights.append(
                decoder_v_u[:, :approx_rank] @ torch.diag(decoder_v_s[:approx_rank]).sqrt()
            )
            w_v_decoder_loraB_weights.append(
                torch.diag(decoder_v_s[:approx_rank]).sqrt() @ decoder_v_v[:approx_rank, :]
            )
            # u : 256*64, V_t: 64*256
            # lora_A = u[:, :approx_rank] @ torch.diag(s[:approx_rank]).sqrt()  # 크기: 256x64
            # lora_B = torch.diag(s[:approx_rank]).sqrt() @ v[:, :approx_rank].T  # 크기: 64x256
    with torch.no_grad():
        for i in range(len_of_layers):
            # base_model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(w_q_encoder_loraA_weights[i].transpose(0, 1))
            # base_model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(w_q_encoder_loraB_weights[i].transpose(0, 1))

            # base_model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(w_v_encoder_loraA_weights[i].transpose(0, 1))
            # base_model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(w_v_encoder_loraB_weights[i].transpose(0, 1))

            # base_model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(w_q_decoder_loraA_weights[i].transpose(0, 1))
            # base_model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(w_q_decoder_loraB_weights[i].transpose(0, 1))

            # base_model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(w_v_decoder_loraA_weights[i].transpose(0, 1))
            # base_model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(w_q_decoder_loraB_weights[i].transpose(0, 1))
            base_model.encoder.layers[i].self_attention.fc_q.lora_A.copy_(w_q_encoder_loraA_weights[i].transpose(0, 1))
            base_model.encoder.layers[i].self_attention.fc_q.lora_B.copy_(w_q_encoder_loraB_weights[i].transpose(0, 1))

            base_model.encoder.layers[i].self_attention.fc_v.lora_A.copy_(w_v_encoder_loraA_weights[i].transpose(0, 1))
            base_model.encoder.layers[i].self_attention.fc_v.lora_B.copy_(w_v_encoder_loraB_weights[i].transpose(0, 1))

            base_model.decoder.layers[i].self_attention.fc_q.lora_A.copy_(w_q_decoder_loraA_weights[i].transpose(0, 1))
            base_model.decoder.layers[i].self_attention.fc_q.lora_B.copy_(w_q_decoder_loraB_weights[i].transpose(0, 1))

            base_model.decoder.layers[i].self_attention.fc_v.lora_A.copy_(w_v_decoder_loraA_weights[i].transpose(0, 1))
            base_model.decoder.layers[i].self_attention.fc_v.lora_B.copy_(w_v_decoder_loraB_weights[i].transpose(0, 1))

            print(
                "model loraA @loraB - 대입 전 계산 결과 값",
                recon_error(
                    base_model.encoder.layers[i].self_attention.fc_q.lora_A.T
                    @ base_model.encoder.layers[i].self_attention.fc_q.lora_B.T,
                    w_q_encoder_loraA_weights[i] @ w_q_encoder_loraB_weights[i],
                ),
            )
            print(
                "svd model encoder q - loraA@loraB",
                recon_error(
                    SVD_model.encoder.layers[i].self_attention.fc_q.weight,
                    w_q_encoder_loraA_weights[i] @ w_q_encoder_loraB_weights[i],
                ),
            )
            print(
                recon_error(
                    SVD_model.decoder.layers[i].self_attention.fc_v.weight,
                    w_v_decoder_loraA_weights[i] @ w_v_decoder_loraB_weights[i],
                )
            )
