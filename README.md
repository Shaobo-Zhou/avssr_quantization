# Audio-Visual Speech Separation and Recognition

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains our multi-task architecture for Audio-Visual Speech Separation and Recognition project.

The architecture is based on [CTCNet](https://arxiv.org/abs/2212.10744), [RTFS-Net](https://openreview.net/forum?id=PEuDO2EiDr) and [Conformer](https://arxiv.org/abs/2005.08100).

## Installation

Follow these steps to work with the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```
### Training

To train the model, run the following command:

```bash
python3 train.py --config_name=CONFIG_NAME # add optional Hydra parameters
```

To use ASR augmentation, add `+model/asr_aug=config_name`.

### Run Inference

To run inference with the quantized model, run the following command:

```bash
python3 inference_quant.py --quant_config_name=YOUR_QUANTIZATION_CONFIG\
```
Additionally, you can set if you want to use percentile clipping and equalization in the ``` --percentile``` and ``` --equalization``` flag, respectively. You can select the quantization scheme to be symmetric/affine with the ``` --qscheme``` flag

Then calculate metrics using the following commands:

```bash
python3  scripts/calculate_metrics.py --dataset_name=YOUR_DATASET --save_name=SPECIAL_SAVE_NAME
```

The metrics will be saved in `data/saved/YOUR_DATASET/SPECIAL_SAVE_NAME_SPLIT_NAME_metric.pth` and printed on the screen.


To calculate MACs (or FLOPs), run:

```bash
python3 flops.py model=MODEL_THAT_YOU_WANT text_encoder.use_lm=False dataloader.batch_size=1
```

### Converting AVSSR StateDict to SS StateDict:

Run the following script:

```bash
python3 scripts/get_ss_state_dict_from_checkpoint.py -c=AVSSR_CHECKPOINT.pth -o=data/pretrain/CTCNET_OR_RTFSNET/NAME.pth
```

### Saving Embeddings for Knowledge Distillation

To save embeddings from pre-trained model, run the following command:

```bash
python3 save_embeddings.py model=MODEL_THAT_YOU_WANT
```

Here is an example for CTCNet:

```bash
python3 save_embeddings.py model=ctcnet +model.ss_pretrain_path="ctcnet/lrs2_best_model.pt" model.ss_model.video_config.shared=False saver.save_key=kd_embedding
```

**Note**: It is also possible to do KD on the fly (if you cannot save tensors due to space limit). To do this, add `+model/ss_teacher=TEACHER_NAME +loss_function.kd_coef=KD_COEF` to your standard `train.py` run.

## Credits

This repository is based on a heavily modified fork of [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
