import torch
from tqdm import tqdm

from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer class. Used to calculate metrics on inference
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        batch_transforms=None,
    ):
        assert (
            config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint"
        self.config = config
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder
        self.beam_size = config.text_encoder.beam_size

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # init model
        self._from_pretrained(config.inferencer.get("from_pretrained"))

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device
        """
        for tensor_for_device in self.config.inferencer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def run_inference(self):
        """
        Save embeddings from each partition
        """
        for part, dataloader in self.evaluation_dataloaders.items():
            self._inference_part(part, dataloader)

    def process_batch(self, batch, batch_idx, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        tokens_logits = batch["tokens_logits"]
        s_audio_length = batch["s_audio_length"]

        argmax_text_list = self.text_encoder.ctc_argmax(tokens_logits, s_audio_length)
        bs_text_list = self.text_encoder.ctc_beam_search(
            tokens_logits, s_audio_length, beam_size=self.beam_size, use_lm=False
        )
        lm_text_list = self.text_encoder.ctc_beam_search(
            tokens_logits, s_audio_length, beam_size=self.beam_size, use_lm=True
        )
        lm_text_small_list = self.text_encoder.ctc_beam_search(
            tokens_logits,
            s_audio_length,
            beam_size=self.beam_size,
            use_lm=False,
            use_lm_small=True,
        )

        batch_size = tokens_logits.shape[0]
        id = batch_idx * batch_size
        for i in range(batch_size):
            mix_audio = batch["mix_audio"][i].detach().cpu()
            s_audio = batch["s_audio"][i].detach().cpu()
            predicted_audio = batch["predicted_audio"][i].detach().cpu()
            argmax_text = argmax_text_list[i]
            bs_text = bs_text_list[i]
            lm_text = lm_text_list[i]
            lm_text_small = lm_text_small_list[i]
            text = batch["s_text"][i]
            output_id = id + i

            output = {
                "mix_audio": mix_audio,
                "s_audio": s_audio,
                "predicted_audio": predicted_audio,
                "argmax_text": argmax_text,
                "bs_text": bs_text,
                "lm_text": lm_text,
                "lm_text_small": lm_text_small,
                "s_text": text,
            }

            torch.save(output, self.save_path / part / f"output_{output_id}.pth")

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions
        """
        self.is_train = False
        self.model.eval()

        # create Save dir
        (self.save_path / part).mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(batch, batch_idx, part)
