import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field
from torchtext import data, datasets
import os
from torchtext.data import Field, BucketIterator

spacy_en = spacy.load("en_core_web_sm")  # 영어 토큰화(tokenization)
spacy_de = spacy.load("de_core_news_sm")  # 독일어 토큰화(tokenization)


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
