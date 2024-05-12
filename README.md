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

2. Install `pre-commit`:

   ```bash
   pre-commit install
   ```

3. Download pretrained models:

   ```bash
   python3 scripts/get_pretrain.py
   ```

4. [LRS2-Mix](https://huggingface.co/datasets/JusperLee/LRS2-2Mix) does not have full audio and text. Download [LRS2 dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) and use these scripts to get audio and text:

   ```bash
   cd scripts
   python3 get_audio_and_text.py --dataset_name=lrs2_rebuild --video_root_path=PATH_TO_RAW_LRS2
   ```

## Experiments

> [!NOTE]
> For pipeline checking, you can download example dataset using `gdown 1ieOhhGkktegV29ct3xB-x69yDm58BENT` and unzip it into `data/example`.

### Training

To train the model, run the following command:

```bash
python3 train.py --config_name=CONFIG_NAME # add optional Hydra parameters
```

To use ASR augmentation, add `+model/asr_aug=config_name`.

### Getting Metrics

To save all predictions, including beam search and its LM version, run the following command:

```bash
python3 inference.py model=YOUR_MODEL\
   datasets=YOUR_DATASET\
   text_encoder.beam_size=BEAM_SIZE\
   +inferencer.from_pretrained=PATH_TO_CHECKPOINT
```

Then calculate metrics using the following commands:

```bash
cd scripts
python3  calculate_metrics.py --dataset_name=YOUR_DATASET
```

The metrics will be saved in `data/saved/YOUR_DATASET/SPLIT_NAME_metric.pth` and printed on the screen.

> [!NOTE]
> Before running `inference.py`, download LM using `scripts/get_lm.py`

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
