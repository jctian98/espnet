#!/bin/bash

# Check if wav.scp file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 wav.scp"
    exit 1
fi

wav_scp=$1
utt2spk_file="utt2spk"
spk2utt_file="spk2utt"
text_file="text"
ref_wav_file="ref_wav.scp"

# Generate utt2spk file
cat $wav_scp | cut -d' ' -f1 | awk -F'-' '{print $0 " " $1}' > $utt2spk_file
cat $wav_scp | cut -d' ' -f1 | awk -F'-' '{print $0 " None"}' > $text_file
cat $wav_scp | cut -d' ' -f1 | awk -F'-' '{print $0 " None"}' > $ref_wav_file

# Generate spk2utt file from utt2spk
cat $utt2spk_file | ../../utils/utt2spk_to_spk2utt.pl > $spk2utt_file

echo "Generated $utt2spk_file and $spk2utt_file"
