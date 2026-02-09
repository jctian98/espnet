(1) clone repo
```bash
git clone -b speechlm3 https://github.com/jctian98/espnet.git espnet_opuslm_demo 
```

(2) build env
```bash
cd espnet_opuslm_demo
bash setup_anaconda.sh miniconda3 opuslm_demo 3.11
source activate_python.sh
pip install editdistance transformers soxr # Install in advance to avoid failure
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
make TH_VERSION=2.4.0 CUDA_VERSION=12.4
```

(3) go to the egss and download checkpoint
```bash
cd ../egs2/librispeech/speechlm1
hf download --repo-type model --local-dir exp/OpusLM_7B_Anneal espnet/OpusLM_7B_Anneal
```

(4) link the test set
```bash
mkdir dump; cd dump
ln -s /work/nvme/bbjs/shared/slm_data/dump/raw_codec_ssl_tts_librispeech . 
cd ..
```

(5) run inference:
```bash
bash tmp.sh
```