#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-sample inference script for 4DThinker.

Usage:
    cd code/train/src
    CUDA_VISIBLE_DEVICES=0 python inference.py \
        --model_path ../../../model/dift/checkpoints \
        --image_dir ./sample_frames \
        --question "Which camera movement technique is employed?"

If --image_dir is not provided, a built-in example is used (requires frames to exist).
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import place_input_image, seed_everything
from task import multiple_input_images_4dthinker_test_preprocess_function


def parse_args():
    parser = argparse.ArgumentParser(description="4DThinker single-sample inference")
    parser.add_argument("--model_path", type=str, default="./model/dift/checkpoints",
                        help="Path to model checkpoint directory")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory containing video frames (frame_0000.jpg, frame_0001.jpg, ...)")
    parser.add_argument("--question", type=str, default=None,
                        help="Question text (with options A/B/C/D)")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="HuggingFace cache directory")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# Default example sample (replace paths with your own frame directory)
DEFAULT_SAMPLE = {
    "image_input": [
        "./data/sample_video/frames/frame_{:04d}.jpg".format(i)
        for i in range(36)
    ],
    "text_input": (
        "Which camera movement technique is employed during the video segment "
        "from 23s to 35s?\n\n"
        "A. Tilt Down\nB. Dolly Out\nC. Dolly In\nD. Tilt Up"
    ),
}


def build_sample(args):
    """Build the sample dict from arguments or use the default."""
    if args.image_dir is not None:
        # Collect all frame_XXXX.jpg files from the directory
        frames = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        if not frames:
            raise FileNotFoundError(
                f"No frame_*.jpg files found in {args.image_dir}"
            )
        question = args.question or DEFAULT_SAMPLE["text_input"]
        return {"image_input": frames, "text_input": question}
    else:
        return DEFAULT_SAMPLE


def main():
    args = parse_args()
    seed_everything(seed=args.seed)
    os.environ["HF_HOME"] = args.cache_dir

    sample = build_sample(args)

    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, cache_dir=args.cache_dir
    )
    # Use single GPU to avoid device mismatch with device_map="auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, device_map={"": 0}, torch_dtype=torch.bfloat16
    )

    processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
    model.eval()

    conversations = multiple_input_images_4dthinker_test_preprocess_function(sample)

    texts = [processor.apply_chat_template(conversations, tokenize=False)]
    texts = [place_input_image(t, sep_token=None) for t in texts]
    image_inputs, _ = process_vision_info(conversations)

    inputs = processor(
        text=[t + "<|im_start|>assistant" for t in texts],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer=processor.tokenizer,
        )

    decoded_output = processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=False
    )
    answer = decoded_output.split("<|im_start|>assistant")[-1]

    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0]
    answer = answer.strip()

    print("\n" + "=" * 60)
    print("Model Output:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
