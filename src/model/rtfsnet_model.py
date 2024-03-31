import numpy as np
import torch

import src.model.asr as asr_module
from src.model.rtfsnet.layers import ConvNormAct
from src.model.rtfsnet.TDAVNet import (
    BaseAVModel,
    RefinementModule,
    decoder,
    encoder,
    mask_generator,
)


class RTFSNetModel(BaseAVModel):
    def __init__(
        self,
        n_src: int,
        enc_dec_params: dict,
        audio_bn_params: dict,
        audio_params: dict,
        mask_generation_params: dict,
        pretrained_vout_chan: int = -1,
        video_bn_params: dict = dict(),
        video_params: dict = dict(),
        fusion_params: dict = dict(),
        print_macs: bool = False,
        # video_model
        video_model=None,
        train_video_model=False,
        # asr_model
        asr_model_name=None,
        asr_model_config=None,
        n_tokens=None,
        **kwargs,
    ):
        super(RTFSNetModel, self).__init__()

        self.n_src = n_src
        self.pretrained_vout_chan = pretrained_vout_chan
        self.audio_bn_params = audio_bn_params
        self.video_bn_params = video_bn_params
        self.enc_dec_params = enc_dec_params
        self.audio_params = audio_params
        self.video_params = video_params
        self.fusion_params = fusion_params
        self.mask_generation_params = mask_generation_params
        self.print_macs = print_macs

        self.encoder: encoder.BaseEncoder = encoder.get(
            self.enc_dec_params["encoder_type"]
        )(
            **self.enc_dec_params,
            in_chan=1,
            upsampling_depth=self.audio_params.get("upsampling_depth", 1),
        )

        self.init_modules()

        self.video_model = video_model
        self.asr_model = getattr(asr_module, asr_model_name)(
            n_tokens=n_tokens, **asr_model_config
        )
        self.train_video_model = train_video_model

    def init_modules(self):
        self.enc_out_chan = self.encoder.get_out_chan()

        self.mask_generation_params[
            "mask_generator_type"
        ] = self.mask_generation_params.get("mask_generator_type", "MaskGenerator")
        self.audio_bn_chan = self.audio_bn_params.get("out_chan", self.enc_out_chan)
        self.audio_bn_params["out_chan"] = self.audio_bn_chan
        self.video_bn_chan = self.video_bn_params.get(
            "out_chan", self.pretrained_vout_chan
        )

        self.audio_bottleneck = ConvNormAct(
            **self.audio_bn_params, in_chan=self.enc_out_chan
        )
        self.video_bottleneck = ConvNormAct(
            **self.video_bn_params, in_chan=self.pretrained_vout_chan
        )

        self.refinement_module = RefinementModule(
            fusion_params=self.fusion_params,
            audio_params=self.audio_params,
            video_params=self.video_params,
            audio_bn_chan=self.audio_bn_chan,
            video_bn_chan=self.video_bn_chan,
        )

        self.mask_generator: mask_generator.BaseMaskGenerator = mask_generator.get(
            self.mask_generation_params["mask_generator_type"]
        )(
            **self.mask_generation_params,
            n_src=self.n_src,
            audio_emb_dim=self.enc_out_chan,
            bottleneck_chan=self.audio_bn_chan,
        )

        self.decoder: decoder.BaseDecoder = decoder.get(
            self.enc_dec_params["decoder_type"]
        )(
            **self.enc_dec_params,
            in_chan=self.enc_out_chan * self.n_src,
            n_src=self.n_src,
        )

        if self.print_macs:
            self.get_MACs()

    def forward(self, mix_audio, s_video, s_audio_length, **batch):
        if self.video_model is None:
            refined_features, audio_mixture_embedding = self.av_forward(mix_audio)
        else:
            if not self.train_video_model:
                with torch.no_grad():
                    mouth_emb = self.video_model(s_video)
            else:
                mouth_emb = self.video_model(s_video)
            refined_features, audio_mixture_embedding = self.av_forward(
                mix_audio, mouth_emb
            )

        # decoder part
        separated_audio_embeddings = self.mask_generator(
            refined_features, audio_mixture_embedding
        )  # B, n_src, N, T, (F)
        separated_audios = self.decoder(
            separated_audio_embeddings, mix_audio.shape
        )  # B, n_src, L

        separated_audios = separated_audios.squeeze(1)  # n_src=1
        # asr part
        tokens_logits, s_audio_length = self.asr_model(refined_features, s_audio_length)

        return {
            "predicted_audio": separated_audios,
            "tokens_logits": tokens_logits,
            "s_audio_length": s_audio_length,
        }

    def av_forward(self, audio_mixture, mouth_embedding):
        audio_mixture_embedding = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, C, T, (F)
        video = self.video_bottleneck(
            mouth_embedding
        )  # B, N2, T2, (F2) -> B, C2, T2, (F2)

        refined_features = self.refinement_module(audio, video)  # B, C, T, (F)

        return refined_features, audio_mixture_embedding

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        full_params = sum([np.prod(p.size()) for p in self.parameters()])
        video_params = sum([np.prod(p.size()) for p in self.video_model.parameters()])
        asr_params = sum([np.prod(p.size()) for p in self.asr_model.parameters()])
        ss_params = full_params - video_params - asr_params

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        result_str = super().__str__()
        result_str = result_str + "\nAll parameters: {}".format(full_params)
        result_str = result_str + "\nVideo parameters: {}".format(video_params)
        result_str = result_str + "\nASR parameters: {}".format(asr_params)
        result_str = result_str + "\nSS parameters: {}".format(ss_params)
        result_str = result_str + "\nTrainable parameters: {}".format(params)

        return result_str

    def get_config(self):
        model_args = {}
        model_args["encoder"] = self.encoder.get_config()
        model_args["audio_bottleneck"] = self.audio_bottleneck.get_config()
        model_args["video_bottleneck"] = self.video_bottleneck.get_config()
        model_args["refinement_module"] = self.refinement_module.get_config()
        model_args["mask_generator"] = self.mask_generator.get_config()
        model_args["decoder"] = self.decoder.get_config()

        return model_args
