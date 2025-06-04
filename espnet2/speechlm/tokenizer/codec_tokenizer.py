#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer


class CodecTokenizer(AbsTokenizer):
    """Codec Tokenizer implementation

    Use cases:
        - use encode and decode for discrete (de)tokenization
        - use encode_continuous and decode_continuous for continuous
          (de)tokenization
        - use forward and detokenization for discrete (de)tokenization
          with flatten sequence style, which is more friendly for
          speechlm task
    """

    def __init__(
        self,
        codec_choice: str,
        codec_fs: int,
        device: str = "cpu",
        dump_audio: bool = False,
        checkpoint_path: str = None,
        config_path: str = None,
        hf_model_tag: str = None,
        max_token_per_frame: int = 8,
    ):
        """Codec Tokenizer initialization

        Each of the codec implementation should assign all following features:
            self.n_codebook (int): the number of codec codebooks.
            self.size_codebook (int): the dimension of codebooks.
            self.sample_rate (int): the sample rate the model trained on.
            self.subsample (int): the subsample rate, a.k.a., frame shift.
        """

        super(CodecTokenizer, self).__init__()
        self.codec_choice = codec_choice
        self.device = device
        self.dump_audio = dump_audio

        if self.codec_choice == "ESPnet":

            if hf_model_tag is not None:
                from espnet2.bin.gan_codec_inference import AudioCoding

                model = AudioCoding.from_pretrained(
                    hf_model_tag, device=str(device)
                ).model
            else:
                from espnet2.tasks.gan_codec import GANCodecTask

                model, _ = GANCodecTask.build_model_from_file(
                    config_path,
                    checkpoint_path,
                    device=str(device),
                )
            self.codec = model

            meta_info = self.codec.meta_info()
            self.n_codebook = min(meta_info["num_streams"], max_token_per_frame)
            self.size_codebook = meta_info["code_size_per_stream"][0]
            self.sample_rate = meta_info["fs"]
            self.subsample = meta_info["frame_shift"]

        elif self.codec_choice == "DAC":
            try:
                import dac
            except ImportError:
                raise ImportError("Install DAC with: pip install descript-audio-codec")

            model_path = dac.utils.download(
                model_type=str(codec_fs).replace("000", "khz")
            )
            self.codec = dac.DAC.load(model_path).to(device)
            self.n_codebook = self.codec.n_codebooks
            self.size_codebook = self.codec.codebook_size
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder_rates)

        elif self.codec_choice == "EnCodec":
            try:
                from encodec import EncodecModel
            except ImportError:
                raise ImportError("Please install Encodec with: pip install -U encodec")

            model_name = "encodec_model_" + str(codec_fs).replace("000", "khz")
            self.codec = getattr(EncodecModel, model_name)().to(device)
            # NOTE (Jinchuan): This Encodec model has 32 codebooks,
            # which is not necessary in usual cases.
            # We only adopt 8 first codebooks, a.k.a., 6kbps.
            bandwidth = 6.0
            self.codec.set_target_bandwidth(bandwidth)
            self.n_codebook = self.codec.quantizer.get_num_quantizers_for_bandwidth(
                self.codec.frame_rate, bandwidth
            )
            self.size_codebook = self.codec.quantizer.bins
            self.sample_rate = self.codec.sample_rate
            self.subsample = np.prod(self.codec.encoder.ratios)

        elif self.codec_choice == "inhouse":
            try:
                from models.soundstream import SoundStream
                from omegaconf import OmegaConf
            except ImportError:
                raise ImportError("fail to use inhouse codec")

            model_path = "encodec_16k_6kbps_multiDisc/ckpt_01135000.pth"
            model_config = "encodec_16k_6kbps_multiDisc/config.yaml"
            config = OmegaConf.load(model_config)
            model = SoundStream(**config.generator.config)

            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict["codec_model"])
            model = model.to(device)
            self.codec = model

            self.n_codebook = 8
            self.sample_rate = 16000
            self.size_codebook = 1024
            self.subsample = 320
        
        elif self.codec_choice == "xcodec":
            try:
                from omegaconf import OmegaConf
                from models.soundstream_semantic import SoundStream
            except:
                raise ImportError(f"X-Codec is not properly detected")
            
            config = OmegaConf.load(config_path)

            model = eval(config.generator.name)(**config.generator.config)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            self.codec = model

            # hard-code to nq = 8
            self.n_codebook = 8
            self.bw = 6
            self.sample_rate = config.generator.config.sample_rate
            self.size_codebook = config.generator.config.bins
            self.subsample = np.prod(config.generator.config.ratios)
            self.audio_norm_scale = config.audio_norm_scale

        elif self.codec_choice == "bigvgan":
            try:
                import json
                from env import AttrDict
                from models import load_generator, apply_generator_forward
                from utils import load_checkpoint, get_linear_and_mel_spectrogram
            except:
                raise ImportError("BigVGAN is not properly detected")

            json_config = json.load(open(config_path))
            h = AttrDict({**json_config})

            model_type = getattr(h, "model_type", "vocoder") # for bigvgan-v2 support on ae branch
            model = load_generator(model_type, h, device)

            checkpoint = load_checkpoint(checkpoint_path, device)
            if "generator" in checkpoint.keys(): # models trained from this repo use "generator" key in checkpoint
                state_dict_load_key = "generator"
            elif "state_dict" in checkpoint.keys(): # models trained from stable-audio-tools use "state_dict" in checkpoint
                state_dict_load_key = "state_dict"
            else:
                raise RuntimeError(f"no valid key found to load from checkpoint.keys(): {checkpoint.keys()}")
            assert state_dict_load_key in checkpoint.keys(),\
                f"{state_dict_load_key} not found in checkpoint.keys() {checkpoint.keys()}"
            model.load_state_dict(checkpoint[state_dict_load_key], strict=False)

            model.eval().requires_grad_(False)
            model.remove_weight_norm()
            model = model.float()

            self.model = model.to(device)

            if h['bottleneck']["type"] == "dithered_fsq":
                self.n_codebook = h['bottleneck']['config']['num_codebooks']
                self.size_codebook = 2048 # hard code to 2k
                self.quantizer_type = "fsq"
            else:
                self.n_codebook = h['bottleneck']['config']['n_codebooks']
                self.size_codebook = h['bottleneck']['config']['codebook_size']
                self.quantizer_type = "rvq"

            self.sample_rate = h['sampling_rate']
            self.subsample = h['hop_size']
            self.normalize_volume = getattr(h, "normalize_volume", True)
            assert getattr(h, 'use_wav_as_input')

        else:
            raise ValueError(f"Codec {codec_choice} is not supported")

    def encode(self, wavs):
        """
        Convert audio waveforms into codec codes
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        """
        assert wavs.dim() == 3 and wavs.size(1) == 1

        if self.codec_choice == "ESPnet":
            codes = self.codec.encode(wavs)
            codes = codes.permute(1, 2, 0)[:, :, : self.n_codebook]

        elif self.codec_choice == "DAC":
            codes = self.codec.encode(wavs)[1]
            codes = codes.transpose(1, 2)

        elif self.codec_choice == "EnCodec":
            encoded_frames = self.codec.encode(wavs)
            codes = encoded_frames[0][0].transpose(1, 2)

        elif self.codec_choice == "inhouse":
            codes = self.codec.encode(wavs).permute(1, 2, 0)
        
        elif self.codec_choice == "xcodec":
            if self.audio_norm_scale < 1.0:
                wavs = wavs * self.audio_norm_scale

            codes = self.codec.encode(wavs, target_bw=self.bw) 
            codes = codes.permute(1, 2, 0)
        
        elif self.codec_choice == "bigvgan":
            if self.normalize_volume:
                # volume normalize, for each sample and channel
                wavs = wavs / (wavs.abs().max(dim=-1, keepdim=True).values + 1e-5) * 0.95

            pad_amount = self.subsample - wavs.size(-1) % self.subsample
            wavs = torch.nn.functional.pad(
                wavs, (0, pad_amount), mode='constant', value=0
            )

            tokens_key = self.model.bottleneck.tokens_id
            codes = self.model.encode(wavs)[tokens_key]
            # NOTE(Jinchuan): output shapes of RVQ and FSQ are different
            if self.quantizer_type == "rvq":
                codes = codes.permute(0, 2, 1)

        else:
            raise NotImplementedError

        return codes

    def encode_continuous(self, wavs):
        """
        Convert audio waveforms into continuous codec encoding results
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            z (torch.Tensor): float tensor in shape [B, T, D]
        """

        if self.codec_choice == "ESPnet":
            z = self.codec.encode_continuous(wavs)
            z = z.transpose(1, 2)

        elif self.codec_choice == "DAC":
            z = self.codec.encode(wavs)[0]
            z = z.transpose(1, 2)

        else:
            raise NotImplementedError

        return z

    def decode(self, codes):
        """
        Recover the waveform from the codes.
        Input:
            codes (torch.Tensor): Int tensor in shape [B, T, n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """

        # NOTE(Jinchuan) The very short input may raise errors, so simply
        # make the output as 0.0
        if codes.size(1) <= 10:
            B, T, _ = codes.size()
            return torch.zeros(
                (B, self.subsample * T),
                dtype=torch.float32,
                device=codes.device,
            )

        if self.codec_choice == "ESPnet":
            codes = codes.permute(2, 0, 1)
            waveform = self.codec.decode(codes).squeeze(1)

        elif self.codec_choice == "DAC":
            z = self.codec.quantizer.from_codes(codes.transpose(1, 2))[0]
            waveform = self.codec.decode(z).squeeze(1)

        elif self.codec_choice == "EnCodec":
            encoded_frames = [(codes.transpose(1, 2), None)]
            waveform = self.codec.decode(encoded_frames).squeeze(1)

        elif self.codec_choice == "inhouse":
            codes = codes.permute(2, 0, 1)
            waveform = self.codec.decode(codes).squeeze(1)
        
        elif self.codec_choice == "xcodec":
            # [B, T, nq] -> [nq, B, T]
            codes = codes.permute(2, 0, 1)
            waveform = self.codec.decode(codes)
        
        elif self.codec_choice == "bigvgan":
            if self.quantizer_type == "rvq":
                codes = codes.permute(0, 2, 1)
            waveform = self.model.decode_tokens(tokens=codes)['decoder_out']
            waveform = torch.clamp(waveform, min=-1.0, max=1.0)
        else:
            raise NotImplementedError

        return waveform

    def decode_continuous(self, z):
        """
        Recover the waveform from the continuous representations of codec
        Input:
            z (torch.Tensor): Float tensor in shape [B, T, D], codec
              continuous representations
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample]
        """
        if self.codec_choice == "ESPnet":
            z = z.transpose(1, 2)
            waveform = self.codec.decode_continuous(z).squeeze(1)

        elif self.codec_choice == "DAC":
            z = z.transpose(1, 2)
            waveform = self.codec.decode(z).squeeze(1)

        else:
            raise NotImplementedError

        return waveform

    def forward(self, wavs):
        """
        Convert audio waveforms into flatten codec codes and resynthesis the audio
        Input:
            wavs (torch.Tensor): float tensor in shape [B, 1, n_sample],
        Output:
            codes (torch.Tensor): Int tensor in shape [B, T * n_codebook],
            resyn_audio (torch.Tensor): float tensor in shape [B, n_samples]
        """
        codes = self.encode(wavs)

        if self.dump_audio:
            resyn_audio = self.decode(codes)
        else:
            resyn_audio = None

        shift = torch.arange(self.n_codebook).to(self.device)
        codes += shift.view(1, 1, -1) * self.size_codebook
        codes = codes.int().flatten(start_dim=1)

        return codes, resyn_audio

    def detokenize(self, codes, n_codebook=None):
        """
        Convert flatten codec codes into resynthesis the audio
        Input:
            codes (torch.Tensor): int tensor in shape [B, T * n_codebook],
                or [T * n_codebook]
        Output:
            waveform (torch.Tensor): float tensor in shape [B, n_sample],
                or [n_sample]
        """

        has_batch = codes.dim() == 2
        if not has_batch:
            codes = codes.unsqueeze(0)

        B, Tnq = codes.size()
        n_codebook = self.n_codebook if n_codebook is None else n_codebook
        assert Tnq % n_codebook == 0, (n_codebook, codes.size())
        codes = codes.view(B, Tnq // n_codebook, n_codebook)

        for l_idx in range(n_codebook):
            codes[:, :, l_idx] -= l_idx * self.size_codebook

        waveform = self.decode(codes)
        if not has_batch:
            waveform = waveform.squeeze(0)

        return waveform


if __name__ == "__main__":
    # a simple use case for batch processing
    device = "cuda:0"
    codec = CodecTokenizer(
        codec_choice="ESPnet",
        codec_fs=16000,
        device=device,
        dump_audio=True,
        checkpoint_path="espnet_codec/16khz_soundstream/train.total_count.best.pth",
        config_path="espnet_codec/16khz_soundstream/config.yaml",
    )

    import soundfile as sf

    waveform, sr = sf.read("1272-128104-0004.wav")
    waveform = (
        torch.from_numpy(waveform).view(1, 1, -1).to(device).float()
    )  # [B, C, n_sample]
    waveform = waveform.repeat(2, 1, 1)

    with torch.no_grad():
        # discrete
        codes = codec.encode(waveform)
        print(f"cdoes: ", codes.size())
        resyn_audio = codec.decode(codes)
        print(f"audio1", resyn_audio.size())
        resyn_audio = resyn_audio[0].cpu().numpy()
        sf.write("resyn1.wav", resyn_audio, sr)

        # continuous
        z = codec.encode_continuous(waveform)
        print(f"z: ", z.size())
        resyn_audio2 = codec.decode_continuous(z)
        print(f"audio2", resyn_audio2.size())
        resyn_audio2 = resyn_audio2[0].cpu().numpy()
        sf.write("resyn2.wav", resyn_audio2, sr)

        # high level API for speechlm
        flatten_codes, _ = codec(waveform)
        print(f"flatten_codes", flatten_codes.size())
        resyn_audio3 = codec.detokenize(flatten_codes)
        print("resyn", resyn_audio3.size())
        resyn_audio3 = resyn_audio3[0].cpu().numpy()
        sf.write("resyn3.wav", resyn_audio3, sr)
