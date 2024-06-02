import multiprocessing
import re
from string import ascii_lowercase

import torch
from pyctcdecode import build_ctcdecoder
from tokenizers import Tokenizer

from src.utils.io_utils import ROOT_PATH


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        use_lm=False,
        use_lm_small=False,
        use_bpe=False,
        nemo_model=None,
        **kwargs
    ):
        """
        :param alphabet: alphabet for language. If None it will be set to ascii
        :param use_lm: whether to enable the support for Language Model Beam Search
        :param use_lm_small: use Small Language Model for Beam Search
        :param use_bpe: Byte Pair Encoding version of encoder
        :param nemo_model: initialize vocabulary using nemo model
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.nemo_model = nemo_model

        if self.nemo_model is not None:
            # import here to avoid long init
            import nemo.collections.asr as nemo_asr

            nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=nemo_model
            )
            self.vocab = nemo_model.decoder.vocabulary

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.use_bpe = use_bpe
        if self.use_bpe:
            tok_path = ROOT_PATH / "data" / "bpe" / "tokenizer.json"
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.char2ind = self.tokenizer.get_vocab()
            self.ind2char = {v: k.lower() for k, v in self.char2ind.items()}
            self.vocab = [self.ind2char[ind] for ind in range(len(self.ind2char))]

        self.bs_decoder = self._create_bs_decoder()

        self.use_lm = use_lm
        if use_lm:
            self.kenlm_path_big = (
                ROOT_PATH / "data" / "lm" / "librispeech" / "4-gram.arpa"
            )
            self.unigram_path_big = self.kenlm_path_big.parent / "librispeech-vocab.txt"
            self.lm_decoder_big = self._create_lm_decoder(
                self.unigram_path_big, self.kenlm_path_big
            )

        self.use_lm = use_lm
        if use_lm:
            self.kenlm_path_small = (
                ROOT_PATH / "data" / "lm" / "librispeech" / "3-gram.pruned.3e-7.arpa"
            )
            self.unigram_path_small = (
                self.kenlm_path_small.parent / "librispeech-vocab.txt"
            )
            self.lm_decoder_small = self._create_lm_decoder(
                self.unigram_path_small, self.kenlm_path_small
            )

    def __len__(self):
        return len(self.vocab)

    def _create_bs_decoder(self):
        """
        Creates beam search ctc_decoder
        """
        vocab = self.vocab

        vocab = [elem.upper() for elem in vocab]

        decoder = build_ctcdecoder(vocab)

        return decoder

    def _create_lm_decoder(self, unigram_path, kenlm_path):
        """
        Creates ctc decoder with the support of Language Model
        """
        vocab = self.vocab

        vocab = [elem.upper() for elem in vocab]

        with unigram_path.open() as f:
            unigram_list = [t for t in f.read().strip().split("\n")]

        decoder = build_ctcdecoder(
            vocab, kenlm_model_path=str(kenlm_path), unigrams=unigram_list
        )

        return decoder

    def encode(self, text):
        """
        Encoding of the text
        :param text: text which will be encoded

        Works for both BPE and usual alphabet
        """
        text = self.normalize_text(text)

        if self.nemo_model is not None:
            # for now, we do not support proper encoding for nemo
            # nemo is used only during inference, so encoding is not useful
            return torch.zeros(len(text))

        if self.use_bpe:
            return torch.tensor(self.tokenizer.encode(text).ids)
        else:
            return torch.tensor([self.char2ind[char] for char in text])

    def ctc_decode(self, inds) -> str:
        """
        CTC decoding: list of int to text
        :param inds: list of indecies (encoded text)
        """
        last_char = self.EMPTY_TOK
        text = []
        for ind in inds:
            if self.ind2char[ind] == last_char:
                continue
            char = self.ind2char[ind]
            if char != self.EMPTY_TOK:
                text.append(char)
            last_char = char
        return "".join(text).strip()

    def ctc_argmax(self, probs: torch.tensor, lengths: torch.tensor):
        """
        Performs argmax ctc decode and returns
        lists of decoded texts.
        :param probs: probabilities from model with shape [N, L, H]
        :param lengths: tensor of shape [N,] containing lengthes without padding
        """

        if self.nemo_model is not None:
            # use beam search with beam size = 1 instead
            # it is equivalent to argmax
            return self.ctc_beam_search(probs, lengths, beam_size=1)

        probs = torch.nn.functional.log_softmax(probs.detach().cpu(), -1)  # to be sure
        logits_list = [probs[i][: lengths[i]].numpy() for i in range(lengths.shape[0])]

        text_list = []
        for logits in logits_list:
            text_list.append(self.ctc_decode(logits.argmax(-1)))
        return text_list

    def ctc_beam_search(
        self,
        probs: torch.tensor,
        lengths: torch.tensor,
        beam_size: int = 100,
        use_lm=False,
        use_lm_small=False,
    ):
        """
        Performs beam search (with or without language model) and returns
        list of decoded texts.
        :param probs: probabilities from model with shape [N, L, H]
        :param lengths: tensor of shape [N,] containing lengthes without padding
        :param beam_size: size of beam to use in decoding
        :param use_lm: use lm version or classic bs
        :param use_lm_small: use small lm instead
        """
        assert not (use_lm and use_lm_small)
        if use_lm:
            decoder = self.lm_decoder_big
        elif use_lm_small:
            decoder = self.lm_decoder_small
        else:
            decoder = self.bs_decoder

        probs = torch.nn.functional.log_softmax(probs.detach().cpu(), -1)  # to be sure
        logits_list = [probs[i][: lengths[i]].numpy() for i in range(lengths.shape[0])]

        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = decoder.decode_batch(pool, logits_list, beam_width=beam_size)

        text_list = [text.strip().lower() for text in text_list]

        if self.nemo_model is not None:
            # remove extra tokens
            text_list = [re.sub(r"[^a-z ]", "", text) for text in text_list]

        return text_list

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
