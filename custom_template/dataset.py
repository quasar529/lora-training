import spacy
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field
from torchtext import data, datasets
import os
from torchtext.data import Field, BucketIterator, TabularDataset


# spacy_en = spacy.load("en_core_web_sm")  # 영어 토큰화(tokenization)
# spacy_de = spacy.load("de_core_news_sm")  # 독일어 토큰화(tokenization)


# 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


class LocalMulti30k(Multi30k):
    dirname = "multi30k"
    name = "data"
    urls = []
    root = "/content/drive/MyDrive/LAB"

    @classmethod
    def _download(cls, *args, **kwargs):
        pass


def prepare_dataset():
    SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
    # 로컬 파일에서 데이터셋 생성
    train_dataset, valid_dataset, test_dataset = LocalMulti30k.splits(
        exts=(".de", ".en"),
        fields=(SRC, TRG),
        root="/content/drive/MyDrive/LAB"
        # root=os.path.join(LocalMulti30k.root, LocalMulti30k.dirname)
    )

    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    return SRC, TRG, train_dataset, valid_dataset, test_dataset


def IMDB():
    batch_size = 30
    max_length = 256

    TEXT = torchtext.data.Field(lower=True, include_lengths=False, batch_first=True)
    LABEL = torchtext.data.Field(sequential=False)
    train_txt, test_txt = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(
        train_txt,
        vectors=torchtext.vocab.GloVe(name="6B", dim=50, max_vectors=50_000),
        max_size=50_000,
    )

    LABEL.build_vocab(train_txt)

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_txt, test_txt),
        batch_size=batch_size,
    )

    return TEXT.vocab.vectors, train_iter, test_iter


def prepare_imdb():
    batch_size = 128

    TEXT = torchtext.data.Field(lower=True, include_lengths=False, batch_first=True, fix_length=100)
    LABEL = torchtext.data.Field(sequential=False)
    # train_txt, test_txt = TabularDataset.splits(
    #     path="/content/drive/MyDrive/LAB/lora-training/custom_template",
    #     train="train_data.csv",
    #     test="test_data.csv",
    #     format="csv",
    #     fields=[("text", TEXT), ("label", LABEL)],
    #     skip_header=True,
    # )
    train_txt, test_txt = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(
        train_txt,
        # vectors=torchtext.vocab.GloVe(name="6B", dim=50, max_vectors=50_000),
        max_size=40000,
        min_freq=5,
    )

    LABEL.build_vocab(train_txt)

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_txt, test_txt), batch_size=batch_size, sort=False, repeat=False
    )
    print(LABEL.vocab.itos)
    print(len(LABEL.vocab))
    print(len(TEXT.vocab))
    for i, batch in enumerate(train_iter):
        src = batch.text
        print(f"첫 번째 배치 크기: (bath size, sentence length) {src.shape}")
        for i in range(src.shape[1]):
            print(f"인덱스 {i}: {src[0][i].item()}")  # 여기에서는 [Seq_num, Seq_len]

        break
    return TEXT, LABEL, train_iter, test_iter


TEXT, LABEL, train_iter, test_iter = prepare_imdb()
