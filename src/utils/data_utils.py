from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def get_dataloaders(config, text_encoder):
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        # dataset partitions init
        dataset = instantiate(config.datasets[dataset_partition],
                              text_encoder=text_encoder)  # instance transorms are defined inside

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
