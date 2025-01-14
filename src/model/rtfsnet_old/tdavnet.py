import torch

from .layers import ConvNormAct
from .TDAVNet import BaseAVModel, RefinementModule, decoder, encoder, mask_generator


class RTFSNetOld(BaseAVModel):
    """
    This is an old version of RTFSNet code provided by authors

    It is needed only to do Knowledge Distillation and load checkpoint.
    Do not use it otherwise.

    Use the same configs as for new RTFSNet, just change the _target_
    """

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
        asr_type: str = "middle",
        *args,
        **kwargs,
    ):
        super().__init__()

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
        self.asr_type = asr_type

        self.encoder: encoder.BaseEncoder = encoder.get(
            self.enc_dec_params["encoder_type"]
        )(
            **self.enc_dec_params,
            in_chan=1,
            upsampling_depth=self.audio_params.get("upsampling_depth", 1),
        )

        self.init_modules()

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
            asr_type=self.asr_type,
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

    def forward(
        self, audio_mixture: torch.Tensor, mouth_embedding: torch.Tensor = None
    ):
        audio_mixture_embedding = self.encoder(audio_mixture)  # B, 1, L -> B, N, T, (F)

        audio = self.audio_bottleneck(audio_mixture_embedding)  # B, C, T, (F)
        video = self.video_bottleneck(
            mouth_embedding
        )  # B, N2, T2, (F2) -> B, C2, T2, (F2)

        refined_features, fused_feats = self.refinement_module(
            audio, video
        )  # B, C, T, (F)

        separated_audio_embeddings = self.mask_generator(
            refined_features, audio_mixture_embedding
        )  # B, n_src, N, T, (F)
        separated_audios = self.decoder(
            separated_audio_embeddings, audio_mixture.shape
        )  # B, n_src, L

        if self.asr_type == "demixed":
            fused_feats = separated_audio_embeddings.squeeze(1)

        predicted_audio = separated_audios.squeeze(1)
        fused_feats = fused_feats.transpose(-1, -2)  # time last
        # fused_feats = refined_features

        if self.asr_type == "audio":
            fused_feats = predicted_audio

        return {
            "predicted_audio": predicted_audio,
            "fused_feats": fused_feats,
            "kd_embedding": refined_features,
        }

    def get_config(self):
        model_args = {}
        model_args["encoder"] = self.encoder.get_config()
        model_args["audio_bottleneck"] = self.audio_bottleneck.get_config()
        model_args["video_bottleneck"] = self.video_bottleneck.get_config()
        model_args["refinement_module"] = self.refinement_module.get_config()
        model_args["mask_generator"] = self.mask_generator.get_config()
        model_args["decoder"] = self.decoder.get_config()

        return model_args

    def init_from(self, path):
        checkpoint = torch.load(path)
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if ".transform_module" in k:
                k = k.replace(".transform_module", "")
                v = v.transpose(0, 1)
            new_state_dict[k] = v
        self.load_state_dict(new_state_dict)
