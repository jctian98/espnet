#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import re

try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
except:
    raise ValueError("Please Install package: pip install pycocoevalcap")

def get_args():
    parser = argparse.ArgumentParser(description="Compute Image Caption Scores")

    # Add required positional argument
    parser.add_argument(
        "--hyp",
        type=str,
        help="The hypothesis text file"
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="The reference text file"
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # Load the file
    hyp = dict()
    for line in open(args.hyp):
        example_id, content = line.strip().split(maxsplit=1)
        content = {"caption": content}
        example_id = example_id.replace("image_to_text_", "")
        example_id = re.sub(r'_sample\d+$', '', example_id)

        # Assume this for MS-COCO dataset, not always correct for others
        img_id, cap_id = example_id.strip().split("-")
        if img_id not in hyp:
            hyp[img_id] = list()
        hyp[img_id].append(content)

    ref = dict()
    for line in open(args.ref):
        example_id, content = line.strip().split(maxsplit=1)
        content = {"caption": content}
        example_id = example_id.replace("image_to_text_", "")
        example_id = re.sub(r'_sample\d+$', '', example_id)

        img_id, cap_id = example_id.strip().split("-")
        if img_id not in ref:
            ref[img_id] = list()
        ref[img_id].append(content)

    # Tokenization
    tokenizer = PTBTokenizer()
    hyp = tokenizer.tokenize(hyp)
    ref = tokenizer.tokenize(ref)

    # make hyp unique
    hyp = {k: list(set(v)) for k, v in hyp.items()}

    # Compute metrics
    scorers = [
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hyp)
        print(f"{method} Score: {score}")


if __name__ == "__main__":
    main()