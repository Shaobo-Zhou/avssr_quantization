import pandas as pd
import torch
from tqdm import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.data_utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                # self.writer.add_scalar(
                #     "learning rate", self.lr_scheduler.get_last_lr()[0]
                # )
                # bug if scheduler
                self.writer.add_scalar(
                    "learning rate", self.optimizer.param_groups[0]["lr"]
                )

                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                self.log_text(batch)  # log train text predictions for the final batch
                self.log_audio(batch)
                break

        log = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(log[self.mnt_metric])

        return log

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self.log_text(batch)  # log predictions for the last batch
            self.log_audio(batch)

        return self.evaluation_metrics.result()

    def log_text(self, batch):
        tokens_logits = batch["tokens_logits"]
        s_audio_length = batch["s_audio_length"]
        target_text = batch["s_text"]
        predicted_argmax_text = self.text_encoder.ctc_argmax(
            tokens_logits, s_audio_length
        )
        predictes_bs_text = self.text_encoder.ctc_beam_search(
            tokens_logits, s_audio_length, beam_size=3, use_lm=False
        )

        df_data = {
            "target_text": target_text,
            "argmax_text": predicted_argmax_text,
            "bs_text_(bs_size=3)": predictes_bs_text,
        }

        df = pd.DataFrame(data=df_data)

        self.writer.add_table("text_predictions", df)

    def log_audio(self, batch, n_log=3):
        mix_audio = batch["mix_audio"]
        s_audio = batch["s_audio"]
        predicted_audio = batch["predicted_audio"]

        for i in range(min(n_log, mix_audio.shape[0])):
            mix = mix_audio[i]
            s = s_audio[i]
            p = predicted_audio[i]
            self.writer.add_audio("mix_audio", mix, sample_rate=16000)
            self.writer.add_audio("source_audio", s, sample_rate=16000)
            self.writer.add_audio("predicted_audio", p, sample_rate=16000)
