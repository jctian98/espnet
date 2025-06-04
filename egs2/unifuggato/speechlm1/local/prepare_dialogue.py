import argparse
import json
import random

from pathlib import Path
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

random.seed(42)

# audio-to-text, generated from claude 4 sonnet
att_prompts = [
    # Direct Action Variants
    "Create text caption from the provided audio input",
    "Produce written description based on the given audio",
    "Generate textual description using the supplied audio",
    "Convert the audio content into corresponding text caption",
    "Transform the provided audio into descriptive text",
    "Render text caption from the supplied audio input",
    "Generate written content according to the audio prompt",
    "Create textual description based on the audio content",
    "Produce text that describes the given audio",
    
    # Instruction-Style Variants
    "Please create text caption that describes the audio",
    "Generate appropriate text description for the given audio",
    "Create matching text caption for this audio input",
    "Write text that represents the provided audio content",
    "Produce text caption that aligns with the audio input",
    "Generate corresponding text based on this audio",
    
    # Process-Focused Variants
    "Process the audio input to generate appropriate text caption",
    "Use the provided audio to create matching text description",
    "Interpret the audio content and generate corresponding text",
    "Analyze the audio input and produce fitting text caption",
    "Convert audio content into textual representation",
    "Transform audio input into written description",
    
    # Result-Oriented Variants
    "Output text caption that reflects the given audio",
    "Deliver written description based on the audio input",
    "Provide textual representation of the audio content",
    "Generate text caption matching the audio specification",
    "Create written description that embodies the audio",
    "Produce textual content as heard in the audio",
    
    # Formal/Technical Variants
    "Execute text generation based on audio parameters provided",
    "Perform audio-to-text conversion using the supplied input",
    "Implement text synthesis according to audio specifications",
    "Generate textual output derived from audio input",
    "Create written representation based on audio parameters",
    
    # Conversational Variants
    "Turn this audio into text caption",
    "Make text description from what's heard in the audio",
    "Create written caption based on what the audio contains",
    "Generate text for this audio input",
    "Produce the text caption describing this audio",
    
    # Transcription/Description Focused
    "Transcribe and describe the provided audio content",
    "Caption the audio with descriptive text",
    "Describe what is heard in the audio using text",
    "Write a caption that summarizes the audio content",
    "Generate descriptive text for the audio clip",
    "Create a written summary of the audio input",
    "Provide text description of the audio's contents",
    "Caption the given audio with appropriate text",
    
    # Analysis-Focused Variants
    "Listen to the audio and generate text caption",
    "Analyze the audio and provide text description",
    "Extract textual description from the audio input",
    "Identify and describe the audio content in text",
    "Recognize and caption the audio with text",
    "Interpret the audio and create text description",
    
    # Quality/Accuracy Focused
    "Generate accurate text caption for the given audio",
    "Create detailed text description based on audio input",
    "Produce comprehensive text caption from the audio",
    "Generate precise textual description of the audio",
    "Create thorough text caption describing the audio",
    
    # Original
    "Generate text caption based on the given audio"
]

# text-to-audio, generated from claude 4 sonnet
tta_prompts = [
    # Direct Action Variants
    "Create audio content from the provided text description",
    "Produce sound based on the textual input provided",
    "Synthesize audio using the given written description",
    "Convert the text description into corresponding audio",
    "Transform the provided text into audio format",
    "Render audio output from the supplied text description",
    "Generate sound content according to the text prompt",
    "Create auditory content based on the descriptive text",
    "Produce audio that matches the given textual description",
    
    # Instruction-Style Variants
    "Please create audio that corresponds to the text description",
    "Synthesize appropriate audio for the given text input",
    "Generate matching audio content for this text description",
    "Create sound that represents the provided written description",
    "Produce audio output that aligns with the text prompt",
    "Generate corresponding audio based on this textual input",
    
    # Process-Focused Variants
    "Process the text description to generate appropriate audio",
    "Use the provided text to create matching audio content",
    "Interpret the text description and generate corresponding sound",
    "Analyze the textual input and produce fitting audio",
    "Convert textual description into auditory representation",
    "Transform written content into audio format",
    
    # Result-Oriented Variants
    "Output audio that reflects the given text description",
    "Deliver sound content based on the textual prompt",
    "Provide audio representation of the described content",
    "Generate audio output matching the text specification",
    "Create sound that embodies the written description",
    "Produce auditory content as described in the text",
    
    # Formal/Technical Variants
    "Execute audio generation based on textual parameters provided",
    "Perform text-to-audio conversion using the supplied description",
    "Implement audio synthesis according to text specifications",
    "Generate acoustic output derived from textual input",
    "Create auditory representation based on descriptive parameters",
    
    # Conversational Variants
    "Turn this text description into audio",
    "Make audio from what's described in the text",
    "Create sound based on what the text describes",
    "Generate audio for this written description",
    "Produce the audio described in this text",
    
    # Original
    "Generate audio based on the given text description"
]

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from AF3')
    
    # Add arguments
    parser.add_argument(
        '--dumpdir',
        type=Path,
        help='The dumpdir of audio tokenization',
    )
    parser.add_argument(
        '--manifests',
        type=Path,
        nargs="+",
        help='All the manifest files',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        help='The name of the dataset',
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['text-to-audio', 'audio-to-text', 'continuous_audio_caption'],
        help='Output directory',
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    for manifest in args.manifests:
        process_one_manifest(manifest, args.dumpdir, args.prefix, args.task)

def process_one_manifest(manifest, dumpdir, prefix, task):
    subset_name = Path(manifest).stem
    if subset_name == "train":
        subset_name = Path(manifest).parent.stem
    
    # read wav.scp
    wav_scp_file = dumpdir / f"{prefix}_{subset_name}" / 'wav.scp'
    if task == "continuous_audio_caption":
        wav_scp_file = str(wav_scp_file).replace("/raw_", "/audio_raw_")
    wav_scp = dict()
    for line in open(wav_scp_file):
        uttid, content = line.strip().split()
        wav_scp[uttid] = content
    
    # define output_dir
    output_dir = dumpdir.parent / f"raw_audio_dialogue_{prefix}" / f"{prefix}_{subset_name}_{task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load manifest
    try:
        json_list = json.load(open(manifest))
    except:
        json_list = [json.loads(line) for line in open(manifest)]
    
    # prompts
    if task == "text-to-audio":
        prompts = tta_prompts
    else:
        prompts = att_prompts

    # parse each example
    dataset = DialogueDataset(task='audio_dialogue')
    for idx, example in enumerate(json_list):
        example_id = f"{prefix}_{subset_name}_{idx}"
        if example_id not in wav_scp:
            print(f"No tokenized audio found for example: {example_id}")
            continue
        
        clip_index = wav_scp[example_id]
        dialogue = Dialogue(task='audio_dialogue')

        # text-audio paired data, without template. E.g., ETTA
        if 'captions' in example:
            # system prompt
            prompt = random.choice(prompts)
            dialogue.add_segment("system", "text_bpe", False, prompt)

            caption = example['captions']
            assert isinstance(caption, str)
            if task == "text-to-audio":
                dialogue.add_segment("user", "text_bpe", False, caption)
                dialogue.add_segment("assistant", "codec", True, clip_index)
            elif task == "continuous_audio_caption":
                dialogue.add_segment("user", "speech_ssl_encoder", False, clip_index)
                dialogue.add_segment("assistant", "text_bpe", True, caption)
            else:
                dialogue.add_segment("user", "codec", False, clip_index)
                dialogue.add_segment("assistant", "text_bpe", True, caption)

        else:
            raise NotImplementedError    
        
        dataset.add_dialogue(example_id, dialogue)
    
    dataset.dump_dataset(output_dir)

if __name__ == "__main__":
    main()