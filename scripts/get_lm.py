import gzip
import os
import shutil
from pathlib import Path

import wget

URL_LINKS = {
    "lm_big": "https://www.openslr.org/resources/11/4-gram.arpa.gz",
    "lm_small": "https://openslr.trmal.net/resources/11/3-gram.pruned.3e-7.arpa.gz",
    "vocab": "https://openslr.trmal.net/resources/11/librispeech-vocab.txt",
}

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def main():
    data_dir = ROOT_PATH / "data" / "lm" / "librispeech"
    data_dir.mkdir(exist_ok=True, parents=True)

    vocab_path = data_dir / "librispeech-vocab.txt"

    arc_path = data_dir / "4-gram.arpa.gz"
    arpa_path = data_dir / "4-gram.arpa"

    if not arpa_path.exists():
        print("Loading lm")
        if not arc_path.exists():
            wget.download(URL_LINKS["lm_big"], out=str(arc_path))
        with gzip.open(str(arc_path), "rb") as f_in:
            with open(str(arpa_path), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(arc_path)

    arc_path = data_dir / "3-gram.pruned.3e-7.arpa.gz"
    arpa_path = data_dir / "3-gram.pruned.3e-7.arpa"

    if not arpa_path.exists():
        print("Loading lm")
        if not arc_path.exists():
            wget.download(URL_LINKS["lm_small"], out=str(arc_path))
        with gzip.open(str(arc_path), "rb") as f_in:
            with open(str(arpa_path), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(arc_path)

    wget.download(URL_LINKS["vocab"], out=str(vocab_path))


if __name__ == "__main__":
    main()
