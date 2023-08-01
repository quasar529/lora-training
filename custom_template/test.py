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
from utils import epoch_time, count_parameters,insert_lora, initialize_weights, W_init_by_lora, W_init_by_SVD, make_W_zero,W_weight_copy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import wandb
import loralib as lora
from train import evaluate

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
    enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load( '/content/drive/MyDrive/LAB/lora-training/custom_template/checkpoints/base_transformer.pt'),strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 뒷 부분의 패딩(padding)에 대해서는 값 무시
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    test_loss = evaluate(model, test_iterator, criterion)
    print("### CHECK THE CONTINUITY ### \n")
    print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
    
if __name__ == "__main__":
    main()