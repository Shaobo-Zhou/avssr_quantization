import argparse
import gzip
import os
import shutil
from pathlib import Path

import wget
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

URL_LINKS = {
    "vocabulary": "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz",
}


def main(args):
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data" / "librispeech"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "librispeech-lm-norm.txt.gz"
    txt_path = data_dir / "librispeech-lm-norm.txt"

    if not txt_path.exists():
        print("Loading vocabulary")
        if not arc_path.exists():
            wget.download(URL_LINKS["vocabulary"], str(arc_path))
        with gzip.open(str(arc_path), "rb") as f_in:
            with open(str(txt_path), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(arc_path)

    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(special_tokens=["", " "], vocab_size=args.vocabulary)
    tokenizer.normalizer = BertNormalizer(strip_accents=True)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train([str(txt_path)], trainer)
    save_path = Path(__file__).absolute().resolve().parent.parent / "data" / "bpe"
    save_path.mkdir(exist_ok=True, parents=True)
    tokenizer.save(str(save_path / "tokenizer.json"))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Tokenizer Script")
    args.add_argument(
        "-v",
        "--vocabulary",
        default=128,
        type=int,
        help="vocabulary size for tokenizer (default 128)",
    )
    main(args.parse_args())
