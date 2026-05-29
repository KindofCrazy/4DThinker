#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSR Benchmark evaluation script: runs inference following inference.py,
extracts content from <answer></answer> tags as model output,
and evaluates using DSR_Suite exact matching. Supports multi-threaded frame extraction.
"""

import os
import sys
import re
import json
import tempfile
import argparse
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

# Ensure DIFT src modules can be imported
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dift", "src")
sys.path.insert(0, SRC_DIR)

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import place_input_image, seed_everything
from task import (
    append_4dthinker_output_format_prompt,
    multiple_input_images_4dthinker_test_preprocess_function,
    VIDEO_FPS,
    VIDEO_MAX_PIXELS,
)

# Default paths
DEFAULT_MODEL_PATH = "./model/dift/checkpoints"
DEFAULT_BENCHMARK_PATH = "./raw_data/DSR_Suite-Data/benchmark.parquet"
DEFAULT_VIDEO_ROOT = "./raw_data/DSR-data/bmk_video"
# Pixel dimensions consistent with task.py training config (VIDEO_MAX_PIXELS=256*28*28)
PROCESSOR_MIN_PIXELS = 28 * 28
PROCESSOR_MAX_PIXELS = VIDEO_MAX_PIXELS  # 256 * 28 * 28

# DSR benchmark `type` column mapped to paper table column names (13 categories in parquet)
SUBTASK_ORDER: List[Tuple[str, str]] = [
    ("abs_dis", "Abs Dis"),
    ("abs_dir", "Abs Dir"),
    ("abs_ori", "Abs Ori"),
    ("abs_spd", "Abs Spd"),
    ("abs_spd_comp", "Abs Spd Comp"),
    ("abs_dir_pred", "Abs Dir Pred"),
    ("rel_dis", "Rel Dis"),
    ("rel_dir", "Rel Dir"),
    ("rel_ori", "Rel Ori"),
    ("rel_spd", "Rel Spd"),
    ("rel_spd_comp", "Rel Spd Comp"),
    ("rel_dir_pred", "Rel Dir Pred"),
    ("non_temp", "Non-Temp Based"),
]
SUBTASK_KEYS = [k for k, _ in SUBTASK_ORDER]


def _normalize_subtask_key(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    return s


def strip_latent_tokens(text: str) -> str:
    """Remove 4dthinker latent placeholders to extract answer from remaining text."""
    if not text:
        return ""
    # Remove <|latent_start|>, <|latent_pad|>, <|latent_end|> and their combinations
    text = re.sub(r"<\|latent_start\|>", " ", text)
    text = re.sub(r"<\|latent_end\|>", " ", text)
    text = re.sub(r"<\|latent_pad\|>", " ", text)
    # Merge extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_answer_from_tags(text: str) -> str:
    """Extract content within <answer></answer> tags from model output."""
    if not text or not isinstance(text, str):
        return ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_option_letter(pred: str) -> str:
    """
    Following DSR_Suite videomme.py extract_characters_regex,
    extract A/B/C/D option letter from prediction text.
    """
    pred = str(pred).strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for prefix in answer_prefixes:
        pred = pred.replace(prefix, "")

    if len(pred.split()) > 10 and not re.search(r"[ABCD]", pred):
        return ""
    m = re.search(r"[ABCD]", pred)
    return m.group(0) if m else ""


def _extract_frames_decord(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    """Extract frames using decord."""
    from decord import VideoReader, cpu

    vid = VideoReader(video_path, ctx=cpu(0))
    video_fps = vid.get_avg_fps()
    n_frames = len(vid)
    total_duration = n_frames / video_fps
    required_frames = max(1, int(total_duration * fps))
    step_size = video_fps / fps if fps > 0 else 1
    indices = [min(int(i * step_size), n_frames - 1) for i in range(required_frames)]
    indices = sorted(set(indices))

    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []
    for i, idx in enumerate(indices):
        frame = vid[idx].asnumpy()
        from PIL import Image

        img = Image.fromarray(frame)
        path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
        img.save(path)
        frame_paths.append(path)
    return frame_paths


def _extract_frames_opencv(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    """Extract frames using OpenCV (fallback when decord fails)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = n_frames / video_fps
    required_frames = max(1, int(total_duration * fps))
    step_size = video_fps / fps if fps > 0 else 1
    indices = [min(int(i * step_size), n_frames - 1) for i in range(required_frames)]
    indices = sorted(set(indices))

    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
        img.save(path)
        frame_paths.append(path)
    cap.release()
    return frame_paths if frame_paths else None


def _extract_frames_ffmpeg(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    """Extract frames using ffmpeg (final fallback, best compatibility)."""
    import shutil
    import subprocess
    import glob

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    target_fps = max(0.1, fps)
    os.makedirs(frame_dir, exist_ok=True)
    out_pattern = os.path.join(frame_dir, "frame_%04d.jpg")
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error", "-i", video_path,
        "-vf", f"fps={target_fps}",
        "-q:v", "2",
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    return frame_paths if frame_paths else None


def extract_frames_from_video(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    """
    Extract frames from video at given fps, save to frame_dir, return frame file paths.
    Priority: decord -> OpenCV -> ffmpeg. Returns None if all methods fail.
    """
    if not os.path.exists(video_path):
        return None

    # 1. Try decord
    try:
        import decord
    except ImportError:
        decord = None

    if decord is not None:
        try:
            return _extract_frames_decord(video_path, fps, frame_dir)
        except Exception as e:
            print(f"[WARN] decord failed, trying OpenCV: {video_path} - {e}")

    # 2. Try OpenCV
    try:
        result = _extract_frames_opencv(video_path, fps, frame_dir)
        if result is not None:
            return result
    except Exception as e:
        print(f"[WARN] OpenCV failed, trying ffmpeg: {video_path} - {e}")

    # 3. Try ffmpeg (best compatibility)
    try:
        result = _extract_frames_ffmpeg(video_path, fps, frame_dir)
        if result is None:
            print(f"[WARN] Frame extraction failed {video_path}: all methods unable to read (video may be corrupted)")
        return result
    except Exception as e:
        print(f"[WARN] Frame extraction failed {video_path}: {e}")
        return None


def load_benchmark(benchmark_path: str) -> List[Dict[str, Any]]:
    """Load benchmark data. Supports parquet and json formats."""
    ext = os.path.splitext(benchmark_path)[1].lower()
    if ext == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(benchmark_path)
        except ImportError:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=benchmark_path, split="train")
            df = ds.to_pandas()
        df = df.reset_index(drop=True)
        df["index"] = range(len(df))
        if "video" not in df.columns and "videoID" in df.columns:
            df["video"] = df["videoID"]
        records = df.to_dict("records")
    elif ext in (".json", ".jsonl"):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                records = [json.loads(line) for line in f]
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else list(data.values())
        for i, r in enumerate(records):
            r["index"] = i
    else:
        raise ValueError(f"Unsupported format: {benchmark_path}")
    return records


def prepare_sample(
    record: Dict,
    video_root: str,
    frame_root: str,
    fps: float,
    video_mode: str = "normal",
) -> Optional[Dict]:
    """
    Prepare a single sample: check video existence, extract frames, build image_input and text_input.
    Returns None if video does not exist (caller will skip, not counted in statistics).
    """
    video_id = record.get("videoID") or record.get("video")
    if not video_id:
        video_id = record.get("video")
    if not video_id:
        return None

    video_path = os.path.join(video_root, f"{video_id}.mp4")
    frame_paths = []
    if video_mode != "text_only":
        if not os.path.exists(video_path):
            return None

        frame_dir = os.path.join(frame_root, video_id)
        frame_paths = extract_frames_from_video(video_path, fps, frame_dir)
        if not frame_paths:
            return None

    question = record.get("question", "")
    options = record.get("options", record.get("candidates", []))
    if options is None:
        options = []
    if isinstance(options, str):
        try:
            options = eval(options)
        except Exception:
            options = []
    if hasattr(options, "tolist"):
        options = options.tolist()
    elif not isinstance(options, list):
        options = list(options) if options else []
    if options:
        question = question + "\n\n" + "\n".join(options)

    st = _normalize_subtask_key(record.get("type", ""))

    return {
        "index": record.get("index", -1),
        "video_id": video_id,
        "video_path": video_path,
        "image_input": frame_paths,
        "text_input": question,
        "answer": record.get("answer", ""),
        "subtask_type": st,
    }


# Aligned with max frames in training data (dift_data.jsonl range 10~50), prevents OOM
MAX_INFERENCE_FRAMES = 50


def run_inference(
    sample: Dict,
    model,
    processor,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    video_mode: str = "normal",
) -> Tuple[str, str]:
    """
    Run inference on a single sample, returns (raw output, extracted answer letter).
    """
    if video_mode == "text_only":
        conversations = [{
            "role": "user",
            "content": [{"type": "text", "text": append_4dthinker_output_format_prompt(sample["text_input"])}],
        }]
        texts = [processor.apply_chat_template(conversations, tokenize=False)]
        inputs = processor(
            text=[t + "<|im_start|>assistant" for t in texts],
            return_tensors="pt",
            padding=True,
        )
    else:
        # Frame limit: uniformly sample long videos to MAX_INFERENCE_FRAMES to prevent OOM
        frames = sample.get("image_input")
        if isinstance(frames, list) and len(frames) > MAX_INFERENCE_FRAMES:
            indices = np.linspace(0, len(frames) - 1, MAX_INFERENCE_FRAMES, dtype=int)
            sample = dict(sample)
            sample["image_input"] = [frames[i] for i in indices]
        if video_mode == "repeat_first":
            frames = sample.get("image_input")
            if isinstance(frames, list) and frames:
                sample = dict(sample)
                sample["image_input"] = [frames[0]] * len(frames)

        conversations = multiple_input_images_4dthinker_test_preprocess_function(sample)
        texts = [processor.apply_chat_template(conversations, tokenize=False)]
        texts = [place_input_image(t, sep_token=None) for t in texts]
        # Consistent with task.py: multi-frame uses video channel, frames in video_inputs (images is None)
        image_inputs, video_inputs = process_vision_info(conversations)

        inputs = processor(
            text=[t + "<|im_start|>assistant" for t in texts],
            images=image_inputs,
            videos=video_inputs,
            videos_kwargs={"fps": VIDEO_FPS},
            return_tensors="pt",
            padding=True,
        )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            tokenizer=processor.tokenizer,
        )

    decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    raw_answer = decoded.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in raw_answer:
        raw_answer = raw_answer.split("<|im_end|>")[0]
    raw_answer = raw_answer.strip()

    # First extract <answer> content, then extract A/B/C/D from it
    answer_content = extract_answer_from_tags(raw_answer)
    pred_letter = extract_option_letter(answer_content)
    if not pred_letter:
        pred_letter = extract_option_letter(raw_answer)
    # 4dthinker output contains many <|latent_*|>, try stripped text if above fails.
    if not pred_letter:
        cleaned = strip_latent_tokens(raw_answer)
        pred_letter = extract_option_letter(cleaned)
    return raw_answer, pred_letter


def aggregate_subtask_metrics(
    records: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    skipped_no_video: List[int],
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """
    Compute per-subtask accuracy across DSR 13 categories; skipped samples (no video) excluded.
    Returns (display_name -> {correct, total, accuracy}), macro_avg (arithmetic mean over categories with samples).
    """
    correct: Dict[str, int] = {k: 0 for k in SUBTASK_KEYS}
    total: Dict[str, int] = {k: 0 for k in SUBTASK_KEYS}

    for r in results:
        k = _normalize_subtask_key(r.get("subtask_type", ""))
        if k not in total:
            continue
        total[k] += 1
        correct[k] += int(r.get("hit", 0))

    by_name: Dict[str, Dict[str, Any]] = {}
    for key, name in SUBTASK_ORDER:
        t = total[key]
        acc = (correct[key] / t) if t > 0 else None
        by_name[name] = {"key": key, "correct": correct[key], "total": t, "accuracy": acc}

    accs = [by_name[n]["accuracy"] for _, n in SUBTASK_ORDER if by_name[n]["total"] > 0]
    macro_avg = sum(accs) / len(accs) if accs else 0.0
    return by_name, macro_avg


def print_subtask_table(by_name: Dict[str, Dict[str, Any]], macro_avg: float) -> None:
    """Print 13-column + Avg table consistent with paper (percentage)."""
    print("\n" + "=" * 50)
    print("Subtask Types (per-category accuracy; skipped samples excluded)")
    print("=" * 50)
    headers = [name for _, name in SUBTASK_ORDER] + ["Avg"]
    # Header
    print(" | ".join(headers))
    print("-" * (len(" | ".join(headers))))
    cells = []
    for _, name in SUBTASK_ORDER:
        d = by_name[name]
        acc = d["accuracy"]
        if acc is None:
            cells.append("N/A")
        else:
            cells.append(f"{acc * 100:.2f}")
    cells.append(f"{macro_avg * 100:.2f}")
    print(" | ".join(cells))
    print("(Values are percentages 0-100; Avg is macro average across subtasks)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="DSR Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model checkpoint path")
    parser.add_argument("--benchmark_path", type=str, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_file", type=str, default=None, help="Results save path, defaults to evaluation/results/")
    parser.add_argument("--frame_root", type=str, default=None, help="Frame cache directory, defaults to temp dir")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of frame extraction threads")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate, for debugging")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="4dthinker outputs latent tokens first, need enough tokens to generate <answer>")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation, e.g. flash_attention_2, sdpa, or eager")
    parser.add_argument("--video_mode", type=str, default="normal", choices=["normal", "text_only", "repeat_first"], help="Ablation mode for visual input")
    parser.add_argument("--latent_size", type=int, default=None, help="Must match training latent_size; defaults to checkpoint config, falls back to 2")
    args = parser.parse_args()

    seed_everything(seed=args.seed)
    os.environ["HF_HOME"] = args.cache_dir

    frame_root = args.frame_root or os.path.join(tempfile.gettempdir(), "dsr_eval_frames")
    os.makedirs(frame_root, exist_ok=True)

    print(f"Loading benchmark: {args.benchmark_path}")
    records = load_benchmark(args.benchmark_path)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"Total {len(records)} samples")

    # Multi-threaded sample preparation (frame extraction)
    samples_to_eval = []
    skipped_no_video = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(
                prepare_sample,
                r,
                args.video_root,
                frame_root,
                VIDEO_FPS,
                args.video_mode,
            ): r
            for r in records
        }
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                sample = fut.result()
                if sample is None:
                    skipped_no_video.append(rec.get("index", -1))
                else:
                    samples_to_eval.append(sample)
            except Exception as e:
                print(f"[WARN] Sample preparation failed index={rec.get('index')}: {e}")
                skipped_no_video.append(rec.get("index", -1))

    print(f"Valid samples: {len(samples_to_eval)}, Skipped (video not found): {len(skipped_no_video)}")

    # Load model
    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, cache_dir=args.cache_dir)
    # Pixel dimensions consistent with task.py training config (VIDEO_MAX_PIXELS=256*28*28)
    processor.image_processor.min_pixels = PROCESSOR_MIN_PIXELS
    processor.image_processor.max_pixels = PROCESSOR_MAX_PIXELS
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
    # Consistent with src/main.py: use config token ids for generation, otherwise may fall to wrong default vocab ids
    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
    model.config.latent_token_id = int(latent_token_idx)
    model.config.latent_start_id = int(latent_start_idx)
    model.config.latent_end_id = int(latent_end_idx)
    if args.latent_size is not None:
        model.config.latent_size = args.latent_size
    elif getattr(model.config, "latent_size", None) is None:
        model.config.latent_size = 2
    model.eval()

    # Inference and evaluation
    results = []
    correct = 0
    total_eval = 0
    skipped_no_video_set = set(skipped_no_video)

    for i, sample in enumerate(samples_to_eval):
        print(f"Sample {i}: {sample}")
        idx = sample["index"]
        gt = str(sample["answer"]).strip().upper()
        if len(gt) > 1:
            gt = gt[0]

        try:
            raw, pred = run_inference(sample, model, processor, args.max_new_tokens, args.temperature, args.top_p, args.video_mode)
            hit = 1 if pred == gt else 0
            correct += hit
            total_eval += 1
            video_path = sample.get("video_path") or os.path.join(args.video_root, f"{sample['video_id']}.mp4")
            result_item = {
                "index": idx,
                "video_id": sample["video_id"],
                "video_path": video_path,
                "question": sample["text_input"],
                "gt": gt,
                "pred": pred,
                "answer": pred,
                "raw": raw,
                "hit": hit,
                "subtask_type": sample.get("subtask_type", ""),
            }
            results.append(result_item)
            # Print each evaluation result
            print(f"\n--- Sample {idx} (video: {sample['video_id']}) ---")
            print(f"Question: {sample['text_input'][:200]}{'...' if len(sample['text_input'])>200 else ''}")
            print(f"Video path: {video_path}")
            print(f"Model output: {raw}")
            print(f"Extracted answer: {pred}")
            print(f"GT answer: {gt}")
            print(f"Correct: {'Y' if hit else 'N'}")
        except Exception as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            print(f"[ERROR] Inference failed index={idx}: {type(e).__name__}: {e!r}")
            traceback.print_exc()
            video_path = sample.get("video_path") or os.path.join(args.video_root, f"{sample.get('video_id', '')}.mp4")
            results.append({
                "index": idx,
                "video_id": sample.get("video_id"),
                "video_path": video_path,
                "question": sample.get("text_input", ""),
                "gt": gt,
                "pred": "",
                "answer": "",
                "raw": str(e),
                "hit": 0,
                "subtask_type": sample.get("subtask_type", ""),
            })
            total_eval += 1
            print(f"GT answer: {gt}")

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(samples_to_eval)}, Current accuracy: {correct/total_eval*100:.2f}%")

    accuracy = correct / total_eval if total_eval > 0 else 0

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total samples: {len(records)}")
    print(f"Evaluated: {total_eval} (skipped {len(skipped_no_video)}, not counted)")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total_eval})")
    print("=" * 50)

    by_subtask, subtask_macro_avg = aggregate_subtask_metrics(records, results, skipped_no_video)
    print_subtask_table(by_subtask, subtask_macro_avg)

    # Save results as .jsonl
    if args.output_file:
        out_path = args.output_file
    else:
        eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(eval_dir, exist_ok=True)
        out_path = os.path.join(eval_dir, f"dsr_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    if not out_path.endswith(".jsonl"):
        out_path = os.path.splitext(out_path)[0] + ".jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved: {out_path}")

    summary_path = os.path.splitext(out_path)[0] + "_summary.json"
    subtask_accuracy = {name: by_subtask[name]["accuracy"] for _, name in SUBTASK_ORDER}
    subtask_correct = {name: by_subtask[name]["correct"] for _, name in SUBTASK_ORDER}
    subtask_total = {name: by_subtask[name]["total"] for _, name in SUBTASK_ORDER}
    summary = {
        "total": len(records),
        "eval_count": total_eval,
        "skipped": len(skipped_no_video),
        "correct": correct,
        "accuracy": accuracy,
        "subtask_accuracy": subtask_accuracy,
        "subtask_correct": subtask_correct,
        "subtask_total": subtask_total,
        "subtask_macro_avg": subtask_macro_avg,
        "subtask_columns_order": [name for _, name in SUBTASK_ORDER] + ["Avg"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
