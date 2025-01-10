import torch
import os
import re
from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import RandomSampler

from src.datasets.collate import collate_fn


class SavedDataLoader(Dataset):
    """
    Custom dataset for loading saved data files
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith('.pth')]
        
        # Sort files numerically based on the number in the filename
        self.files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        #print("File loading order:", self.files)  # Debug: Print the sorted order

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.files[idx])
        data = torch.load(file_path)
        return data

def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def get_dataloaders(config, text_encoder):
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)

    # dataloaders init
    dataloaders = {}
    #total_size=0
    for dataset_partition in config.datasets.keys():
        # dataset partitions init
        dataset = instantiate(
            config.datasets[dataset_partition], text_encoder=text_encoder
        )  # instance transorms are defined inside

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
        )
        #total_size+=len(dataset)
        dataloaders[dataset_partition] = partition_dataloader
    #print(f"Total size of the dataset: {total_size}")

    return dataloaders, batch_transforms

def get_calibration_samples_for_ASR(config, text_encoder, load_path, num_calibration_samples):
    data_path = load_path/f'test'
    dataset = SavedDataLoader(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    calibration_samples = []
    for i, sample in enumerate(dataloader):
        if i >= num_calibration_samples:
            break
        calibration_samples.append(sample)
    
    return calibration_samples



def get_calibration_samples(config, text_encoder, num_calibration_samples):
    # dataset partitions init
    #dataset = instantiate(config.datasets['test'], text_encoder=text_encoder)
    dataset = instantiate(config.datasets['val'], text_encoder=text_encoder)
    # Create DataLoader with a fixed batch size for calibration
    """ sampler = RandomSampler(dataset, replacement=False)

    # Use Subset to get a subset of the dataset with the specified number of samples
    subset_indices = list(sampler)[:num_calibration_samples]
    subset = Subset(dataset, subset_indices)
    print(subset_indices)

    # Create DataLoader for the subset
    dataloader = DataLoader(
        subset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,  # No need to shuffle, as indices are already random
    ) """
    dataloader = DataLoader(
        dataset,
        batch_size=32,  
        collate_fn=collate_fn,
        shuffle=True,  # Shuffle to get random calibration samples
    )

    calibration_samples = []
    for i, sample in enumerate(dataloader):
        if i >= num_calibration_samples:
            break
        calibration_samples.append(sample)
    
    return calibration_samples