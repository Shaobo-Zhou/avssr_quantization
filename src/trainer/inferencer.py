import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import os
import re
from src.model.asr import ASRNemo
from quantization import QuantizedASRModel
from quantization_brevitas import QuantizedBrevModel
from src.trainer.base_trainer import BaseTrainer


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
        load_path=None,
        batch_transforms=None,
        SS_targets = False,
        quantization = False
    ):
        self.config = config
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder
        self.beam_size = config.text_encoder.beam_size
        self.quantization = quantization
        self.SS_targets = SS_targets
        if self.quantization == True: 
            self.device = "cpu"

        # define dataloaders
        #self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.evaluation_dataloaders = {'test': dataloaders['test']}

        self.load_path = load_path
        self.save_path = save_path

        # init model
        # for asr nemo, init ss using ss_pretrain_path
        if not ((isinstance(model.asr_model, ASRNemo)) or (isinstance(model, QuantizedASRModel)) or (isinstance(model, QuantizedBrevModel))):
            assert (
                config.inferencer.get("from_pretrained") is not None
            ), "Provide checkpoint"
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device
        """
        for tensor_for_device in self.config.inferencer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def run_inference(self, max_samples=None):
        """
        Save embeddings from each partition
        """
        start_time = time.time()  
        for part, dataloader in self.evaluation_dataloaders.items():
            self._inference_part(part, dataloader, max_samples)

        end_time = time.time()  # End the timer
        inference_time = end_time - start_time
        
        print(f"Inference time: {inference_time:.4f} seconds")
    def process_saved_data(self, part, max_samples=None):
        """
        Process previously saved data files in the specified directory.
        """
        print(f'Saving at {self.save_path}')
        data_path = self.load_path / part
        dataset = SavedDataLoader(data_path)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        sample_count = 0
        # Process each batch in the saved data
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc=part,
            total=len(dataloader),
        ):
            
            batch = self.transform_batch(batch)  
            fused_feats = batch['fused_feats'].to(self.device)
            s_audio_length = batch["s_audio_length"].squeeze().to(self.device)

            sample_count += batch['mix_audio'].size(0)

            if self.quantization == True:
                outputs = self.model.asr_model(input_signal=fused_feats, input_signal_length=s_audio_length)
                tokens_logits = outputs[0]
            else: 
                outputs = self.model.asr_model(fused_feats, s_audio_length, None)
                batch.update(outputs)
                tokens_logits = batch["tokens_logits"]

            argmax_text_list = self.text_encoder.ctc_argmax(tokens_logits, s_audio_length)
            bs_text_list = self.text_encoder.ctc_beam_search(
                tokens_logits, s_audio_length, beam_size=self.beam_size, use_lm=False
            )

            batch_size = batch['s_audio'].shape[0]
            id = batch_idx * batch_size
            for i in range(batch_size):
                mix_audio = batch["mix_audio"][i].detach().cpu()
                s_audio = batch["s_audio"][i].detach().cpu()
                predicted_audio = batch["predicted_audio"][i].detach().cpu()
                argmax_text = argmax_text_list[i]
                bs_text = bs_text_list[i]
                text = batch["s_text"][i]
                output_id = id + i
                output = {
                "mix_audio": mix_audio,
                "s_audio": s_audio,
                "predicted_audio": predicted_audio,
                "argmax_text": argmax_text,
                "bs_text": bs_text,
                "s_text": text,
                }
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")
            if max_samples is not None and sample_count >= max_samples:
                print(f"Reached maximum number of samples: {max_samples}")
                break

    def save_SS_output(self, batch, batch_idx, part):
        """
        Save the output of the SS model
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch) 
        outputs = self.model(**batch, skip_ASR=True)
        batch.update(outputs)
        s_audio_length = batch["s_audio_length"]
        batch_size = batch['s_audio'].shape[0]
        id = batch_idx * batch_size
        for i in range(batch_size):
            mix_audio = batch["mix_audio"][i].detach().cpu()
            s_audio = batch["s_audio"][i].detach().cpu()
            predicted_audio = batch["predicted_audio"][i].detach().cpu()
            fused_feats = batch["fused_feats"][i].detach().cpu()
            s_text = batch["s_text"][i]
            output_id = id + i

            output = {
                "mix_audio": mix_audio,
                "s_audio": s_audio,
                "predicted_audio": predicted_audio,
                "s_audio_length": s_audio_length,
                "s_text": s_text,
                "fused_feats": fused_feats
            }

            torch.save(output, self.save_path / part / f"output_{output_id}.pth")

    def process_batch(self, batch, batch_idx, part):
        
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  
        outputs = self.model(**batch)
        batch.update(outputs)

        tokens_logits = batch["tokens_logits"]
        s_audio_length = batch["s_audio_length"]

        argmax_text_list = self.text_encoder.ctc_argmax(tokens_logits, s_audio_length)
        bs_text_list = self.text_encoder.ctc_beam_search(
            tokens_logits, s_audio_length, beam_size=self.beam_size, use_lm=False
        ) 
        s_audio_length = batch["s_audio_length"]
        batch_size = batch['s_audio'].shape[0]
        id = batch_idx * batch_size
        for i in range(batch_size):
            mix_audio = batch["mix_audio"][i].detach().cpu()
            s_audio = batch["s_audio"][i].detach().cpu()
            predicted_audio = batch["predicted_audio"][i].detach().cpu()
            fused_feats = batch["fused_feats"][i].detach().cpu()
            argmax_text = argmax_text_list[i]
            bs_text = bs_text_list[i] 
            text = batch["s_text"][i]
            output_id = id + i

            output = {
                "mix_audio": mix_audio,
                "s_audio": s_audio,
                "predicted_audio": predicted_audio,
                "s_audio_length": s_audio_length,
                "fused_feats": fused_feats,
                "s_text": text,
                "argmax_text": argmax_text,
                "bs_text": bs_text,    
            }
            torch.save(output, self.save_path / part / f"output_{output_id}.pth")

    def _inference_part(self, part, dataloader, max_samples=None):
        """
        Run inference on a given partition and save predictions
        """
        self.is_train = False
        self.model.eval()

        # create Save dir
        (self.save_path / part).mkdir(exist_ok=True, parents=True)
        
        sample_count=0

        with torch.no_grad():
            if self.load_path is not None: 
                self.process_saved_data(part, max_samples=max_samples)
            else:
                for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
                ):
                    
                    # Increment sample count based on batch size
                    sample_count += batch['mix_audio'].size(0)
                    if self.SS_targets:
                        print('Saving SS outputs')
                        self.save_SS_output(batch, batch_idx, part)
                    else:
                        batch = self.process_batch(batch, batch_idx, part)
                    
                    # Break when max_samples is reached, if specified
                    if max_samples is not None and sample_count >= max_samples:
                        print(f"Reached maximum number of samples: {max_samples}")
                        break

