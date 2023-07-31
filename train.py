import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from torchtext.datasets import Multi30k
from torchtext.data import Field
from torchtext import data, datasets
from torchtext.data import Field, BucketIterator
import spacy
import os

from model.model import Transformer, Encoder, Decoder
from utils.util import epoch_time, count_parameters, initialize_weights

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


### CUSTOM DATASET
class LocalMulti30k(Multi30k):
    dirname = "multi30k"
    name = "data"
    urls = []
    root = "/content/drive/MyDrive/LAB"

    @classmethod
    def _download(cls, *args, **kwargs):
        pass


spacy_en = spacy.load("en_core_web_sm")  # 영어 토큰화(tokenization)
spacy_de = spacy.load("de_core_news_sm")  # 독일어 토큰화(tokenization)
# spacy_en = spacy.load("en")  # 영어 토큰화(tokenization)
# spacy_de = spacy.load("de")  # 독일어 토큰화(tokenization)

tokenized = spacy_en.tokenizer("I am a graduate student.")


# 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


# 로컬 파일에서 데이터셋 생성
SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
train_dataset, valid_dataset, test_dataset = LocalMulti30k.splits(
    exts=(".de", ".en"),
    fields=(SRC, TRG),
    root="/content/drive/MyDrive/LAB"
    # root=os.path.join(LocalMulti30k.root, LocalMulti30k.dirname)
)
SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)
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
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]


def main(config):
    logger = config.get_logger("train")
    device, device_ids = prepare_device(config["n_gpu"])
    enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
    # build model architecture, then print to console
    # model = config.init_obj("arch", module_arch)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.apply(initialize_weights)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    logger.info(model)

    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset), batch_size=128, device=device
    )

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=train_iterator,
        name=config["name"],
        valid_data_loader=valid_iterator,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
