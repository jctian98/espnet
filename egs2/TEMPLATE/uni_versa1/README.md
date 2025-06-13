# Universa: Unified Versatile Evaluation for Speech & Audio Metrics

This recipe implements **Universa**, a unified framework for evaluating speech and audio quality using multi-task metric prediction. Universa learns from raw audio, and optionally reference waveform and reference text, to predict diverse evaluation metrics such as MOS, speaker similarity, WER, and more.

---

## 🛠️ Supported Use Cases

- Learning from audio + optional reference text/wav
- Multi-metric prediction using a shared or expert-specialized encoder
- Utterance-level and system-level scoring
- Easy extension to new metrics or input modalities

---

## 🧩 Prerequisites

Ensure ESPnet2 is installed and activated:

```bash
cd espnet/tools
./setup_anaconda.sh anaconda espnet
cd ..
pip install -e .
```

## 🗃️ Data Preparation

The Universa recipe expects each dataset (train, valid, test) to follow Kaldi-style directory conventions with some required and optional files:

### Required files:
- `wav.scp`: mapping from utterance IDs to audio file paths
- `metric.scp`: mapping from utterance IDs to reference metric scores

### Optional files:
- `ref_wav.scp`: reference audio file path (used for metrics like speaker similarity or distortion)
- `text`: reference transcription (used for text-conditioned metrics like WER, BERTScore, etc.)
- `metric2id`: mapping from metric names (e.g., `mos`, `wer`) to integer class IDs (auto-generated if not provided)
- `ref_segments`: segment info for reference audio (if needed)
- `utt2num_samples`: will be auto-generated to check duration filtering

To prepare and preprocess the datasets, use the following command:

```bash
./run.sh \
  --train_set train_set \
  --valid_set valid_set \
  --test_sets test_set \
  --fs 16000 \
  --feats_type raw \
  --use_ref_wav true \
  --use_ref_text true \
  --stage 1 --stop_stage 5
```

## 🏋️ Training
Train a Universa model using your preferred feature extractor:

```bash
./run.sh \
  --train_set train_set \
  --valid_set valid_set \
  --test_sets test_set \
  --train_config conf/train_universa_hubert.yaml \
  --ngpu 1 \
  --use_ref_wav true \
  --use_ref_text true \
  --metric2type mos \
  --stage 6 --stop_stage 7
```

## 🔍 Inference
To decode test sets using a trained model:
```bash
./run.sh \
  --skip_data_prep true \
  --skip_train true \
  --inference_config conf/decode.yaml \
  --inference_model valid.loss.best.pth \
  --stage 8 --stop_stage 8
```

## Evaluation
Evaluate predicted metrics against references:
```bash
./run.sh \
  --skip_data_prep true \
  --skip_train true \
  --skip_eval false \
  --sys_info path/to/system_info.tsv \
  --stage 9
```
You will obtain:
- `utt_result.json`: utterance-level evalaution results
- `sys_result.json`: system-level evalaution (if `sys_info` is provided)

## 📁 Directory Structure

```bash
egs2/universa1/
├── conf/
│   ├── train_universa_hubert.yaml
│   ├── train_universa_mrhubert.yaml
│   ├── train_universa_wavlm.yaml
│   └── decode.yaml
├── local/
│   └── data.sh               # Your corpus-specific data prep
├── run.sh                    # Main training/inference script
└── data/                     # Contains wav.scp, metric.scp, etc.
```

## ☁️ Hugging Face Upload (Optional)
After training and evaluation, package and upload your model:

```bash
./run.sh \
  --stage 10 --stop_stage 11 \
  --hf_repo your-username/your-model-name
```
Ensure `git-lfs` is installed and that you have permission to push to the repo.

## Notes
- Reference audio/text are optional but improve prediction performance
- `metric2id` maps metric names (e.g., `mos`, `wer`) to numerical targets
- `sys_info` should be a tab-separated file with `utt_id<TAB>system_id`

Example `sys_info.tsv`
```bash
utt_001    System_A
utt_002    System_A
utt_003    System_B
```

## 📬 Contact

This recipe was developed by [Jiatong Shi](https://github.com/ftshijt) as part of the ESPnet project.

For questions, bug reports, or contributions, please:

- Open an issue in the [ESPnet GitHub repository](https://github.com/espnet/espnet/issues)
- Submit a pull request with improvements
- Reach out via GitHub discussions or tag the author in relevant issues
