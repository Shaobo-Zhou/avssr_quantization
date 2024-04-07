# NeMo Forced Aligner (NFA)

> [!NOTE]
> Alignment code is taken from [NeMo GitHub](https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner).

To align the text run the following code:

```bash
# create manifest.json for chosen DATASET
python3 create_manifest.py \
	--dataset_name=DATASET_NAME \
	--audio_path=AUDIO_PATH_IN_DATASET_DIR \
	--text_path=TEXT_PATH_IN_DATASET_DIR

# use NeMo FA to align audio with text
# for pretrained_name any NeMo ASR CTC or Hybrid model can be used
# we suggest using the largest model: stt_en_fastconformer_ctc_xxlarge
python3 align.py \
	pretrained_name=NeMo_ASR_MODEL_NAME \
  	manifest_filepath=PATH_TO_DATASET_DIR/alignment/manifest.json \
  	output_dir=PATH_TO_DATASET_DIR/alignment/nfa_output/ \
  	additional_segment_grouping_separator="|" \
  	ass_file_config.vertical_alignment="bottom" \
  	ass_file_config.text_already_spoken_rgb=[66,245,212] \
  	ass_file_config.text_being_spoken_rgb=[242,222,44] \
  	ass_file_config.text_not_yet_spoken_rgb=[223,242,239] \
  	ctm_file_config.remove_blank_tokens=True

# crop text to SECONDS
python3 crop_text.py --dataset_name=DATASET_NAME --seconds=SECONDS
```

---

<p align="center">
Try it out: <a href="https://huggingface.co/spaces/erastorgueva-nv/NeMo-Forced-Aligner">HuggingFace Space 🎤</a> | Tutorial: <a href="https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb">"How to use NFA?" 🚀</a> | Blog post: <a href="https://nvidia.github.io/NeMo/blogs/2023/2023-08-forced-alignment/">"How does forced alignment work?" 📚</a>
</p>

<p align="center">
<img width="80%" src="https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/nfa_forced_alignment_pipeline.png">
</p>

NFA is a tool for generating token-, word- and segment-level timestamps of speech in audio using NeMo's CTC-based Automatic Speech Recognition models. You can provide your own reference text, or use ASR-generated transcription. You can use NeMo's ASR Model checkpoints out of the box in [14+ languages](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages), or train your own model. NFA can be used on long audio files of 1+ hours duration (subject to your hardware and the ASR model used).

## Quickstart

1. Install [NeMo](https://github.com/NVIDIA/NeMo#installation).
2. Prepare a NeMo-style manifest containing the paths of audio files you would like to process, and (optionally) their text.
3. Run NFA's `align.py` script with the desired config, e.g.:
   ```bash
   python <path_to_NeMo>/tools/nemo_forced_aligner/align.py \
       pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
       manifest_filepath=<path to manifest of utterances you want to align> \
       output_dir=<path to where your output files will be saved>
   ```

<p align="center">
	<img src="https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/nfa_run.png">
</p>

## Documentation

More documentation is available [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/nemo_forced_aligner.html).
