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

### Inference

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

### Converting to onnx:

Run the following script:

```bash
python3 to_onnx.py --quant_config_name=YOUR_QUANTIZATION_CONFIG\
```

### Converting to TensorRT engine

Before converting to TensorRT: Go through the following steps to change the quantization scheme to symmetric quantization with INT8 if it is not already done (Pytorch enforces affine quantization on activations using UINT8 representation):
1. Run: 
```bash
python3 jetson_scripts/force_quant_sym.py model_path=PATH_TO_YOUR_MODEL
```
2. Run: 
```bash
python3 jetson_scripts/fix_dequant.py model_path=PATH_TO_YOUR_MODEL
```
3. Use onnxsim to simplify the ONNX model and improve compatibility with TensorRT. It is recommended to run this tool both before and after the previous steps:
``` onnxsim PATH_TO_YOUR_MODEL SIMPLIFIED_MODEL_PATH```


Use the following command:

```bash
path/to/trtexec --onnx=YOUR_MODEL.onnx --verbose --int8 --allowGPUFallback --saveEngine=YOUR_ENGINE_PATH
```
### Running Inference with TensorRT engine

Use the following command:

```bash
path/to/trtexec --loadEngine=YOUR_ENGINE_PATH --verbose --int8 --allowGPUFallback --iterations=N
```

This will pass random samples as input, to use true samples, specify with `` --loadInputs``

### Running Inference in Python on Jetson

Use the following command:

```bash
python3 pipeline.py --engine_path=/path/to/ASR_engine \
                       --ss_model_path=/path/to/ss_model.onnx \
                       --output_folder=/path/to/output
```

### Collect hardware metrics

You can observe hardware metrics by installing the ``jtop`` package 

For TensorRT, ``trtexec``command also provides metrics such as latency
## Credits

This repository is based on a heavily modified fork of [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
