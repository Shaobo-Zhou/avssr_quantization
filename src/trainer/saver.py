import torch
from tqdm import tqdm

from src.trainer.base_trainer import BaseTrainer


class Saver(BaseTrainer):
    """
    Saver class. Used to save embeddings from pre-trained models
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_key,
        save_path,
        id_key,
        batch_transforms=None,
    ):
        self.config = config
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_key = save_key
        self.save_path = save_path
        self.id_key = id_key

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device
        """
        for tensor_for_device in self.config.saver.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def save(self):
        """
        Save embeddings from each partition
        """
        for part, dataloader in self.evaluation_dataloaders.items():
            self._save_part(part, dataloader)

    def process_batch(self, batch):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        embedding_batch = batch[self.save_key].detach().cpu()
        id_batch = batch[self.id_key]

        for embedding, id in zip(embedding_batch, id_batch):
            save_path = self.save_path / f"{id}.pt"
            torch.save(embedding, save_path)

    def _save_part(self, part, dataloader):
        """
        Save embeddings from the given partition
        """
        self.is_train = False
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(batch)
