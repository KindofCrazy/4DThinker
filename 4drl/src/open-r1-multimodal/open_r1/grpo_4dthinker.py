# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
4DThinker GRPO RL training script.

Aligns with SFT training (train.sh / src/main.py) and evaluation (dsr_eval.py):
  - Qwen2.5-VL model
  - Video frames sampled at fps=1 (VIDEO_FPS=1), max_pixels=256*28*28
  - Special latent tokens: <|latent_pad|>, <|latent_start|>, <|latent_end|>
  - 4dthinker prompt suffix (identical to task.py FOUR_D_THINKER_TEXT_INPUT_SUFFIX)
  - process_vision_info with fps kwarg for video (mirrors dsr_eval.py inference)
  - skip_special_tokens=False so latent tokens appear in completions
  - Reward: extract <answer>...</answer> → A/B/C/D letter → compare to gt

Dataset (rl_data.jsonl) fields:
  - video_path  : absolute path to .mp4
  - Question    : question text
  - A/B/C/D     : option texts
  - Correct     : ground-truth letter (A/B/C/D)
"""

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any

import PIL.Image
import torch
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
from transformers.utils.versions import require_version

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.data_utils import is_conversational
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from accelerate.utils import is_peft_model

from open_r1.vlm_modules import Qwen2VLModule
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load,
)

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

logger = logging.get_logger(__name__)

# ============================================================
# Constants – must match task.py / dsr_eval.py
# ============================================================
VIDEO_FPS = 1
VIDEO_MAX_PIXELS = 256 * 28 * 28   # 200704
PROCESSOR_MIN_PIXELS = 28 * 28     # 784

# Identical to task.py FOUR_D_THINKER_TEXT_INPUT_SUFFIX
FOUR_D_THINKER_TEXT_INPUT_SUFFIX = (
    "\nFirst, think about the reasoning process with a mental image of the relevant object or region, "
    "then provide the user with the answer. "
    "Put the reasoning in </think>...</think> and only the final answer in <answer>...</answer>, "
    "e.g. </think> reasoning process (with mental imagery as needed) </think><answer> answer here </answer>."
)

# Special tokens added in src/main.py
LATENT_TOKENS = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]


# ============================================================
# Script arguments
# ============================================================
@dataclass
class GRPOScriptArguments(ScriptArguments):
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Path(s) to JSONL data files, separated by ':'"},
    )
    max_pixels: Optional[int] = field(
        default=VIDEO_MAX_PIXELS,
        metadata={"help": "Maximum pixels for video frames"},
    )
    min_pixels: Optional[int] = field(
        default=PROCESSOR_MIN_PIXELS,
        metadata={"help": "Minimum pixels for video frames"},
    )
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "Reward functions: 'accuracy' and/or 'format'"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Compat param, unused for Qwen2.5-VL"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


# ============================================================
# Frame extraction helpers (mirrors dsr_eval.py)
# ============================================================

def _extract_frames_decord(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
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
        img = PIL.Image.fromarray(frame)
        path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
        img.save(path)
        frame_paths.append(path)
    return frame_paths


def _extract_frames_opencv(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    import cv2

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
        img = PIL.Image.fromarray(frame_rgb)
        path = os.path.join(frame_dir, f"frame_{i:04d}.jpg")
        img.save(path)
        frame_paths.append(path)
    cap.release()
    return frame_paths if frame_paths else None


def _extract_frames_ffmpeg(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    import shutil
    import subprocess
    import glob

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    os.makedirs(frame_dir, exist_ok=True)
    out_pattern = os.path.join(frame_dir, "frame_%04d.jpg")
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error", "-i", video_path,
        "-vf", f"fps={max(0.1, fps)}",
        "-q:v", "2",
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except Exception:
        return None

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    return frame_paths if frame_paths else None


def extract_frames_from_video(video_path: str, fps: float, frame_dir: str) -> Optional[List[str]]:
    """Extract frames at fps from video, save to frame_dir. Returns list of paths or None."""
    if not os.path.exists(video_path):
        return None

    # Cache check: if frame_dir already has frames, reuse them
    import glob as _glob
    if os.path.isdir(frame_dir):
        existing = sorted(_glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
        if existing:
            return existing

    try:
        import decord  # noqa
        has_decord = True
    except ImportError:
        has_decord = False

    if has_decord:
        try:
            return _extract_frames_decord(video_path, fps, frame_dir)
        except Exception as e:
            logger.warning(f"decord failed for {video_path}: {e}")

    try:
        result = _extract_frames_opencv(video_path, fps, frame_dir)
        if result is not None:
            return result
    except Exception as e:
        logger.warning(f"OpenCV failed for {video_path}: {e}")

    try:
        result = _extract_frames_ffmpeg(video_path, fps, frame_dir)
        if result is None:
            logger.warning(f"All frame extraction methods failed for {video_path}")
        return result
    except Exception as e:
        logger.warning(f"ffmpeg failed for {video_path}: {e}")
        return None


# ============================================================
# Answer extraction helpers (mirrors dsr_eval.py)
# ============================================================


def extract_answer_from_tags(text: str) -> str:
    """Extract content inside <answer>...</answer>."""
    if not text:
        return ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def extract_option_letter(pred: str) -> str:
    """Extract A/B/C/D letter from prediction text (mirrors dsr_eval.py)."""
    pred = str(pred).strip()
    for prefix in [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is",
        "Best answer:", "Best option:", "Answer:", "Option:",
    ]:
        pred = pred.replace(prefix, "")
    if len(pred.split()) > 10 and not re.search(r"[A-F]", pred):
        return ""
    m = re.search(r"[A-F]", pred)
    return m.group(0) if m else ""


# ============================================================
# Reward functions
# ============================================================

def accuracy_reward(completions, solution, **kwargs):
    """
    Accuracy reward: extract A/B/C/D from <answer>...</answer> and compare to gt.
    Prints the full model output for each completion (as required).
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        gt = str(sol).strip().upper()
        if len(gt) > 1:
            gt = gt[0]

        # Extract predicted answer: <answer> tags → A/B/C/D
        answer_content = extract_answer_from_tags(content)
        pred = extract_option_letter(answer_content)

        if not pred:
            pred = extract_option_letter(content)

        reward = 1.0 if (pred and pred == gt) else 0.0
        rewards.append(reward)

        # Strip trailing endoftext padding for cleaner logs
        _clean = re.sub(r'(<\|endoftext\|>)+', ' [PAD...]', content)
        print(f"[ACCURACY_REWARD] GT={gt} | Pred={pred} | Reward={reward} | output={_clean[:300]}")

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"GT: {gt}\n")
                    f.write(f"Pred: {pred}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards


def format_reward(completions, **kwargs):
    """
    Format reward: checks <think>…</think><answer>…</answer> structure.
    Latent tokens are stripped before matching so they don't break the check.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content in completion_contents:
        match = re.search(pattern, content, re.DOTALL)
        reward = 1.0 if match else 0.0
        rewards.append(reward)

        _clean_fmt = re.sub(r'(<\|endoftext\|>)+', ' [PAD...]', content)
        print(f"[FORMAT_REWARD] Reward={reward} | snippet: {_clean_fmt[:200]}")

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                fmt_log = log_path.replace(".txt", "_format.txt")
                with open(fmt_log, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")

    return rewards


REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


# ============================================================
# Dataset loading
# ============================================================

def build_question_text(item: dict) -> str:
    """
    Build full question text from JSONL item.
    Mirrors dsr_eval.py prepare_sample: append options then 4dthinker suffix.
    """
    question = item.get("Question", "")
    options = []
    for letter in ["A", "B", "C", "D"]:
        val = item.get(letter, "")
        if val:
            options.append(f"{letter}. {val}")
    if options:
        question = question + "\n\n" + "\n".join(options)
    return question + FOUR_D_THINKER_TEXT_INPUT_SUFFIX


def load_4dthinker_dataset(data_file_paths: str, frame_cache_dir: str) -> Dataset:
    """
    Load rl_data.jsonl, extract video frames at fps=1, build prompt conversations.

    Each example:
      prompt   : list[dict]  – user turn with {"type":"video","video":[frame_paths],"fps":1}
      solution : str         – correct letter ("A"/"B"/"C"/"D")
      video_path: str
    """
    import json

    os.makedirs(frame_cache_dir, exist_ok=True)

    _rank = os.environ.get('LOCAL_RANK', '?')

    raw_data = []
    for data_file in data_file_paths.split(":"):
        print(f"[RANK {_rank}] Loading {data_file} ...", flush=True)
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
    print(f"[RANK {_rank}] Loaded {len(raw_data)} raw examples", flush=True)

    processed = []
    skipped = 0
    total = len(raw_data)
    log_interval = max(1, total // 20)  # log every 5%
    for idx, item in enumerate(raw_data):
        if idx % log_interval == 0 or idx == total - 1:
            print(f"[RANK {_rank}] Processing: {idx+1}/{total} ({100*(idx+1)/total:.0f}%), ok={len(processed)}, skip={skipped}", flush=True)

        # Support two input modes:
        #   1) "video_path" → extract frames from video file
        #   2) "frame_paths" → pre-extracted frame image paths (from SFT data)
        pre_frames = item.get("frame_paths", None)
        video_path = item.get("video_path", "")

        if pre_frames and isinstance(pre_frames, list) and len(pre_frames) > 0:
            # Pre-extracted frames: verify first frame exists
            if not os.path.exists(pre_frames[0]):
                skipped += 1
                continue
            frame_paths = pre_frames
        elif video_path and os.path.exists(video_path):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join(frame_cache_dir, video_id)
            frame_paths = extract_frames_from_video(video_path, VIDEO_FPS, frame_dir)
            if not frame_paths:
                skipped += 1
                continue
        else:
            skipped += 1
            continue

        correct = str(item.get("Correct", "")).strip().upper()
        if not correct:
            skipped += 1
            continue

        question_text = build_question_text(item)

        # Conversation format mirrors dsr_eval.py / task.py:
        # video content (list of frame file paths) + text question
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_paths,  # list of frame image paths
                        "fps": VIDEO_FPS,
                    },
                    {
                        "type": "text",
                        "text": question_text,
                    },
                ],
            }
        ]

        processed.append({
            "prompt": prompt,
            "solution": correct,
            "video_path": video_path or (pre_frames[0] if pre_frames else ""),
        })

    logger.info(f"Processed {len(processed)} examples, skipped {skipped}")
    if processed:
        logger.info(f"First example:\n  solution: {processed[0]['solution']}\n  video_path: {processed[0]['video_path']}\n  prompt: {processed[0]['prompt']}")
    return Dataset.from_list(processed)


# ============================================================
# Custom GRPO Trainer for 4DThinker video inputs
# ============================================================

class FourDThinkerGRPOTrainer(Trainer):
    """
    GRPO trainer adapted for 4DThinker:
      - Handles video inputs (list of frame paths) via process_vision_info
      - Adds latent special tokens to processor / model
      - Uses skip_special_tokens=False to preserve latent tokens in completions
      - Reward kwargs pass `solution` column as `solution`
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs,
        args: GRPOConfig = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        peft_config=None,
        freeze_vision_modules: bool = False,
        attn_implementation: str = "flash_attention_2",
        max_pixels: int = VIDEO_MAX_PIXELS,
        min_pixels: int = PROCESSOR_MIN_PIXELS,
        **kwargs,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{model_name.split('/')[-1]}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = "bfloat16"
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        assert isinstance(model, str), "model must be a string path"
        model_id = model

        # Load model
        logger.info(f"Loading model from {model_id} ...")
        loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **model_init_kwargs
        )
        # When gradient_checkpointing=True, use_cache is set to False above for training.
        # But generation (inference) still needs KV cache for the latent mode in _sample.
        # Reset generation_config.use_cache to True so _prepare_generation_config
        # doesn't override the trainer's use_cache=True setting.
        loaded_model.generation_config.use_cache = True
        logger.info("Model loaded.")

        # LoRA
        if peft_config is not None:
            from peft import get_peft_model
            vision_keywords = ["visual"]

            def _find_linear_names(m, skip_keywords):
                cls = torch.nn.Linear
                names = set()
                for name, mod in m.named_modules():
                    if any(kw in name for kw in skip_keywords):
                        continue
                    if isinstance(mod, cls) and "embed_tokens" not in name:
                        names.add(name)
                return list(names)

            peft_config.target_modules = _find_linear_names(loaded_model, vision_keywords)
            loaded_model = get_peft_model(loaded_model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            for n, p in loaded_model.named_parameters():
                if "visual" in n:
                    p.requires_grad = False

        trainable = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        if args.gradient_checkpointing:
            loaded_model.gradient_checkpointing_enable()

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            logger.info("beta=0.0, skipping reference model.")
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            logger.info("Loading reference model (ZeRO-3 mode) ...")
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **model_init_kwargs
            )
            logger.info("Reference model loaded.")
        elif is_peft_model(loaded_model):
            logger.info("PEFT model detected, skipping reference model.")
            self.ref_model = None
        else:
            logger.info("Creating reference model (deep copy) ...")
            self.ref_model = create_reference_model(loaded_model)
            logger.info("Reference model created.")

        # Processor
        logger.info("Loading processor ...")
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            processing_class.image_processor.max_pixels = max_pixels
            processing_class.image_processor.min_pixels = min_pixels

        # Add latent special tokens (same as src/main.py)
        for tok in LATENT_TOKENS:
            processing_class.tokenizer.add_tokens(tok, special_tokens=True)

        latent_pad_id = int(processing_class.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0])
        latent_start_id = int(processing_class.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0])
        latent_end_id = int(processing_class.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0])

        loaded_model.config.latent_token_id = latent_pad_id
        loaded_model.config.latent_start_id = latent_start_id
        loaded_model.config.latent_end_id = latent_end_id
        loaded_model.resize_token_embeddings(len(processing_class.tokenizer))

        if self.ref_model is not None:
            self.ref_model.config.latent_token_id = latent_pad_id
            self.ref_model.config.latent_start_id = latent_start_id
            self.ref_model.config.latent_end_id = latent_end_id
            self.ref_model.resize_token_embeddings(len(processing_class.tokenizer))

        processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.reward_processing_classes = [None] * len(reward_funcs)
        self.reward_weights = args.reward_weights or [1.0] * len(reward_funcs)

        # Generation config
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=processing_class.tokenizer.pad_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            output_hidden_states=True,
            use_cache=True,
        )

        self.num_generations = args.num_generations
        self.num_iterations = getattr(args, "num_iterations", 1)
        self.processing_class = processing_class

        from collections import defaultdict
        self._metrics = defaultdict(list)

        # Custom collator: pass through raw dicts as a list (prompt is a nested dict,
        # default_data_collator can't tensorize it).
        def _passthrough_collator(features):
            return features

        logger.info("Calling Trainer.__init__ (DeepSpeed engine setup) ...")
        super().__init__(
            model=loaded_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=_passthrough_collator,
            **kwargs,
        )
        logger.info("Trainer initialization complete.")

        # Place ref_model on the correct device (DeepSpeed only wraps the training model,
        # not self.ref_model — without this, ref_model stays on CPU and crashes).
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                if is_deepspeed_zero3_enabled():
                    # ZeRO-3: must shard ref_model across GPUs via DeepSpeed engine
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
                else:
                    # ZeRO-2/1: ref_model fits on each GPU, just move it there.
                    # Cannot use prepare_deepspeed because DeepSpeed engine tries to
                    # create an optimizer, which fails on fully-frozen models (empty param groups).
                    self.ref_model = self.ref_model.to(self.accelerator.device)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _precompute_inputs_embeds(self, model, prompt_ids_B, completion_ids_BG,
                                  prompt_mask_B, mm_B, G):
        """Pre-compute inputs_embeds with video tokens embedded for B unique prompts,
        then expand to B*G.  This avoids running the vision encoder G times per prompt
        during the grad-enabled training forward.

        The vision encoder runs under torch.no_grad() (its output is deterministic and
        identical for all G generations of the same prompt).  The text embed_tokens call
        keeps gradients enabled so the training loss can back-propagate.

        Returns inputs_embeds [B*G, seq, hidden] and rope_mm dict (for RoPE).
        """
        unwrapped = self.accelerator.unwrap_model(model)
        B = prompt_ids_B.size(0)

        # --- Vision encoder (no grad, B unique prompts only) ---
        video_embeds = None
        image_embeds = None
        with torch.no_grad():
            if "pixel_values_videos" in mm_B and mm_B["pixel_values_videos"] is not None:
                pv = mm_B["pixel_values_videos"].type(unwrapped.visual.dtype)
                vg = mm_B["video_grid_thw"]
                video_embeds = unwrapped.visual(pv, grid_thw=vg)  # [total_feats, H]

                n_video_tokens = (prompt_ids_B == unwrapped.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features/tokens mismatch: {n_video_features} vs {n_video_tokens}"
                    )

            if "pixel_values" in mm_B and mm_B["pixel_values"] is not None:
                pv_img = mm_B["pixel_values"].type(unwrapped.visual.dtype)
                ig = mm_B["image_grid_thw"]
                image_embeds = unwrapped.visual(pv_img, grid_thw=ig)

                n_image_tokens = (prompt_ids_B == unwrapped.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features/tokens mismatch: {n_image_features} vs {n_image_tokens}"
                    )

        # --- Text embeddings (with grad for prompt part) ---
        # Prompt embeddings need grad so loss back-propagates through embed_tokens
        prompt_embeds_B = unwrapped.model.embed_tokens(prompt_ids_B)  # [B, P, H]

        # Scatter vision embeddings into prompt positions
        if video_embeds is not None:
            mask = prompt_ids_B == unwrapped.config.video_token_id
            if mask.any():
                mask_exp = mask.unsqueeze(-1).expand_as(prompt_embeds_B)
                video_embeds = video_embeds.to(prompt_embeds_B.device, prompt_embeds_B.dtype)
                prompt_embeds_B = prompt_embeds_B.masked_scatter(mask_exp, video_embeds)

        if image_embeds is not None:
            mask = prompt_ids_B == unwrapped.config.image_token_id
            if mask.any():
                mask_exp = mask.unsqueeze(-1).expand_as(prompt_embeds_B)
                image_embeds = image_embeds.to(prompt_embeds_B.device, prompt_embeds_B.dtype)
                prompt_embeds_B = prompt_embeds_B.masked_scatter(mask_exp, image_embeds)

        # Expand prompt B → B*G (identical prompt repeated for each generation)
        prompt_embeds_BG = prompt_embeds_B.repeat_interleave(G, dim=0)  # [B*G, P, H]

        # Completion embeddings: each generation has different tokens (with grad)
        comp_embeds_BG = unwrapped.model.embed_tokens(completion_ids_BG)  # [B*G, C, H]

        inputs_embeds_BG = torch.cat([prompt_embeds_BG, comp_embeds_BG], dim=1)  # [B*G, P+C, H]

        # Return grid_thw expanded for RoPE calculation
        rope_mm = {}
        if "video_grid_thw" in mm_B:
            rope_mm["video_grid_thw"] = mm_B["video_grid_thw"].repeat_interleave(G, dim=0)
        if "image_grid_thw" in mm_B:
            rope_mm["image_grid_thw"] = mm_B["image_grid_thw"].repeat_interleave(G, dim=0)

        return inputs_embeds_BG, rope_mm

    def _get_per_token_logps(self, model, input_ids, attention_mask,
                             inputs_embeds=None, prompt_length=0,
                             **multimodal_inputs):
        """Compute per-token log probabilities.

        Two code-paths depending on whether gradients are needed:
        - Grad-enabled  (training model in compute_loss):
              Single forward pass — DeepSpeed ZeRO-2 requires exactly one
              forward/backward per step; multiple chunk-forwards would trigger
              "parameter already reduced" assertion.
              Uses F.cross_entropy (fused kernel) to avoid materialising a full
              [BG, seq, vocab] log-softmax tensor.
              When inputs_embeds is provided, the vision encoder is skipped.
              When prompt_length > 0, only the completion portion of logits is
              kept (prompt logits are discarded to save ~6x memory).
        - No-grad  (ref / old model):
              Batch-chunked forward (CHUNK=4) to cap peak VRAM on the logits tensor.
        """
        import torch.nn.functional as _F

        BG = input_ids.size(0)
        mm = {k: v for k, v in multimodal_inputs.items() if v is not None}

        # ================================================================
        # Grad-enabled path — single forward (for DeepSpeed ZeRO-2 safety)
        # ================================================================
        if torch.is_grad_enabled():
            fwd_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, **mm)
            if inputs_embeds is not None:
                # Pre-computed embeddings: skip vision encoder inside model
                fwd_kwargs["inputs_embeds"] = inputs_embeds
            # skip_lm_head=True: model returns hidden_states without materializing
            # full [BG, seq, vocab] logits (~37 GiB float32 for seq=4100).
            # We apply lm_head per-sample on completion-only tokens (~0.6 GiB each).
            fwd_kwargs["skip_lm_head"] = True
            outputs = model(**fwd_kwargs)

            unwrapped = self.accelerator.unwrap_model(model)
            if prompt_length > 0:
                hs = outputs.hidden_states[:, prompt_length - 1:-1]  # [BG, comp, H]
                del outputs
                labels = input_ids[:, prompt_length:]     # [BG, comp]
            else:
                hs = outputs.hidden_states[:, :-1]        # [BG, seq-1, H]
                del outputs
                labels = input_ids[:, 1:]

            # Per-sample lm_head + CE with gradient checkpointing: recomputes
            # logits during backward so only one [comp, vocab] tensor is alive
            # at a time (~0.6 GiB) instead of all 16 (~9.3 GiB).
            from torch.utils.checkpoint import checkpoint as _ckpt

            def _lm_head_ce(hs_i, labels_i):
                logits_i = unwrapped.lm_head(hs_i).float()  # [comp, vocab]
                return -_F.cross_entropy(logits_i, labels_i, reduction="none")

            parts = []
            for i in range(BG):
                chunk_logps = _ckpt(
                    _lm_head_ce, hs[i], labels[i], use_reentrant=False,
                )
                parts.append(chunk_logps.unsqueeze(0))
            del hs
            per_token_logps = torch.cat(parts, dim=0)  # [BG, comp]
            return per_token_logps

        # ================================================================
        # No-grad path — batch-chunked forward to avoid OOM
        # ================================================================
        CHUNK = max(1, min(4, BG))

        if BG <= CHUNK:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **mm)
            logits = outputs.logits[:, :-1]
            labels = input_ids[:, 1:]
            per_token_logps = -_F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(BG, -1)
            return per_token_logps

        # Precompute per-item patch counts for pixel_values slicing
        # NOTE: pixel_values_videos has shape [total_patches, C] where total_patches = sum(t*h*w)
        # The spatial merge happens INSIDE the visual encoder, so raw patch count = t * h * w
        video_grid_thw = multimodal_inputs.get("video_grid_thw")
        pixel_values_videos = multimodal_inputs.get("pixel_values_videos")

        patches_per_item = None
        if video_grid_thw is not None and pixel_values_videos is not None:
            patches_per_item = (
                video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
            ).tolist()

        image_grid_thw = multimodal_inputs.get("image_grid_thw")
        pixel_values = multimodal_inputs.get("pixel_values")
        img_patches_per_item = None
        if image_grid_thw is not None and pixel_values is not None:
            img_patches_per_item = (
                image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
            ).tolist()

        all_logps = []
        vid_offset = 0
        img_offset = 0

        for start in range(0, BG, CHUNK):
            end = min(start + CHUNK, BG)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            chunk_sz = end - start

            chunk_mm = {}
            if patches_per_item is not None:
                chunk_grid = video_grid_thw[start:end]
                chunk_patches = sum(patches_per_item[start:end])
                chunk_mm["pixel_values_videos"] = pixel_values_videos[vid_offset:vid_offset + chunk_patches]
                chunk_mm["video_grid_thw"] = chunk_grid
                vid_offset += chunk_patches

            if img_patches_per_item is not None:
                chunk_img_grid = image_grid_thw[start:end]
                chunk_img_patches = sum(img_patches_per_item[start:end])
                chunk_mm["pixel_values"] = pixel_values[img_offset:img_offset + chunk_img_patches]
                chunk_mm["image_grid_thw"] = chunk_img_grid
                img_offset += chunk_img_patches

            outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask, **chunk_mm)
            logits = outputs.logits[:, :-1]
            labels = chunk_ids[:, 1:]
            per_token_logps = -_F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(chunk_sz, -1)
            all_logps.append(per_token_logps)
            del outputs, logits

        return torch.cat(all_logps, dim=0)

    @staticmethod
    def _extract_vision_from_prompt(prompt, max_frames=16, max_side=448):
        """Extract images/videos from a conversation prompt (replaces qwen_vl_utils.process_vision_info).

        Args:
            max_frames: cap number of frames to avoid OOM (must be even for temporal_patch_size=2).
            max_side: resize frames so the longer side <= max_side, reducing memory.
        """
        import numpy as np
        images, videos = [], []
        for msg in prompt:
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            for item in content:
                if item.get("type") == "image":
                    img = PIL.Image.open(item["image"]).convert("RGB")
                    images.append(img)
                elif item.get("type") == "video":
                    frame_paths = item.get("video", [])
                    # Subsample frames if too many
                    if len(frame_paths) > max_frames:
                        indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
                        frame_paths = [frame_paths[i] for i in indices]
                    frames = []
                    for p in frame_paths:
                        img = PIL.Image.open(p).convert("RGB")
                        # Resize to limit memory
                        w, h = img.size
                        if max(w, h) > max_side:
                            scale = max_side / max(w, h)
                            img = img.resize((int(w * scale), int(h * scale)), PIL.Image.BILINEAR)
                        frames.append(np.array(img))
                    if frames:
                        videos.append(np.stack(frames, axis=0))
        return images or None, videos or None

    def _generate_and_score_completions(self, inputs, model):
        """
        Core GRPO step: for B prompts, generate G completions each in a single
        batched generate() call (batch_size=B*G).  Then compute rewards and
        group-normalised advantages.
        """
        device = self.accelerator.device
        G = self.num_generations
        B = len(inputs)
        BG = B * G

        # --- Clean prompts (strip None keys injected by Arrow) ---
        prompts = [x["prompt"] for x in inputs]
        cleaned_prompts = []
        for p in prompts:
            cleaned_msgs = []
            for msg in p:
                content = msg.get("content", [])
                if isinstance(content, list):
                    cleaned_content = [
                        {k: v for k, v in item.items() if v is not None}
                        for item in content
                    ]
                    cleaned_msgs.append({**msg, "content": cleaned_content})
                else:
                    cleaned_msgs.append(msg)
            cleaned_prompts.append(cleaned_msgs)
        prompts = cleaned_prompts

        # --- Process each prompt separately ---
        all_prompt_ids = []   # list of [1, P_i]
        all_mm = []           # list of dicts
        prompt_lengths = []
        pad_id = self.processing_class.tokenizer.pad_token_id or 0

        for b in range(B):
            prompt = prompts[b]
            text = self.processing_class.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            img_i, vid_i = self._extract_vision_from_prompt(prompt)
            kw = dict(text=[text], return_tensors="pt", padding=False)
            if img_i:
                kw["images"] = img_i
            if vid_i:
                kw["videos"] = vid_i
                kw["videos_kwargs"] = {"fps": VIDEO_FPS}
            sample_input = self.processing_class(**kw)

            p_ids = sample_input["input_ids"].to(device)  # [1, P_i]
            all_prompt_ids.append(p_ids)
            prompt_lengths.append(p_ids.size(1))

            mm = {}
            for key in ("pixel_values", "pixel_values_videos",
                         "image_grid_thw", "video_grid_thw"):
                if key in sample_input:
                    mm[key] = sample_input[key].to(device)
            all_mm.append(mm)

        # --- DEBUG ---
        for b in range(B):
            vpath = inputs[b].get("video_path", "N/A")
            sol = inputs[b].get("solution", "N/A")
            q_text = ""
            for msg in inputs[b].get("prompt", []):
                for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else []):
                    if item.get("type") == "text":
                        q_text = item.get("text", "")[:200]
            print(f"[DEBUG-INPUT] b={b} video={vpath} | solution={sol} | question={q_text}", flush=True)

        # --- Left-pad prompts to max length for batched generation ---
        max_prompt_len = max(prompt_lengths)
        padded_ids_list = []
        padded_mask_list = []
        for p_ids in all_prompt_ids:
            p_len = p_ids.size(1)
            pad_len = max_prompt_len - p_len
            if pad_len > 0:
                pad_t = torch.full((1, pad_len), pad_id, dtype=p_ids.dtype, device=device)
                p_ids = torch.cat([pad_t, p_ids], dim=1)
                mask = torch.cat([
                    torch.zeros(1, pad_len, dtype=torch.long, device=device),
                    torch.ones(1, p_len, dtype=torch.long, device=device),
                ], dim=1)
            else:
                mask = torch.ones(1, p_len, dtype=torch.long, device=device)
            padded_ids_list.append(p_ids)
            padded_mask_list.append(mask)

        prompt_ids_B = torch.cat(padded_ids_list, dim=0)    # [B, max_P]
        prompt_mask_B = torch.cat(padded_mask_list, dim=0)   # [B, max_P]

        # --- Expand to B*G for generation ---
        # repeat_interleave keeps same-prompt copies adjacent: [p0,p0,..,p0, p1,p1,..,p1, ...]
        prompt_ids_BG = prompt_ids_B.repeat_interleave(G, dim=0)   # [B*G, max_P]
        prompt_mask_BG = prompt_mask_B.repeat_interleave(G, dim=0) # [B*G, max_P]

        # Expand multimodal inputs: each prompt's mm repeated G times, then concatenated
        gen_kwargs = {}
        for key in ("pixel_values", "pixel_values_videos",
                     "image_grid_thw", "video_grid_thw"):
            parts = []
            for b in range(B):
                if key in all_mm[b]:
                    t = all_mm[b][key]
                    parts.append(t.repeat(G, *([1] * (t.dim() - 1))))
            if parts:
                gen_kwargs[key] = torch.cat(parts, dim=0)
        gen_kwargs["input_ids"] = prompt_ids_BG
        gen_kwargs["attention_mask"] = prompt_mask_BG

        # --- Generate B*G completions in one call ---
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.eval()
            unwrapped_model.gradient_checkpointing_disable()

            print(f"[DEBUG-GEN] Generating B={B} x G={G} = {BG} completions in one batched call ...", flush=True)

            with torch.no_grad():
                output_ids = unwrapped_model.generate(
                    **gen_kwargs,
                    generation_config=self.generation_config,
                )  # [B*G, max_P + max_C]

            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.train()

        # --- Extract completions (all prompts left-padded to max_prompt_len) ---
        completion_ids = output_ids[:, max_prompt_len:]          # [B*G, max_C]
        max_comp_len = completion_ids.size(1)

        # --- Completion mask (up to & including first EOS) ---
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((BG,), max_comp_len, dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(max_comp_len, device=device).unsqueeze(0).expand(BG, -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # Prompt ids/mask for forward pass (same left-padded version, expanded to B*G)
        prompt_ids = prompt_ids_BG                                # [B*G, max_P]
        prompt_mask = prompt_mask_BG                              # [B*G, max_P]

        # Full attention mask for forward pass (includes latent tokens)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Mask latent tokens out of the *loss* mask
        latent_token_ids = {
            self.model.config.latent_token_id,
            self.model.config.latent_start_id,
            self.model.config.latent_end_id,
        }
        for tid in latent_token_ids:
            completion_mask = completion_mask * (completion_ids != tid).int()
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # --- Multimodal inputs for forward pass (same expansion as generation) ---
        multimodal_inputs = {}
        for key in ("pixel_values", "pixel_values_videos",
                     "image_grid_thw", "video_grid_thw"):
            if key in gen_kwargs and key not in ("input_ids", "attention_mask"):
                multimodal_inputs[key] = gen_kwargs[key]

        # --- B-sized (non-expanded) multimodal inputs for pre-computing inputs_embeds ---
        multimodal_inputs_B = {}
        for key in ("pixel_values", "pixel_values_videos",
                     "image_grid_thw", "video_grid_thw"):
            parts = []
            for b in range(B):
                if key in all_mm[b]:
                    parts.append(all_mm[b][key])
            if parts:
                multimodal_inputs_B[key] = torch.cat(parts, dim=0)

        # --- Ref / old log-probs (batched forward, no grad) ---
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs,
                )
                old_per_token_logps = old_per_token_logps[:, max_prompt_len - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    **multimodal_inputs,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask,
                        **multimodal_inputs,
                    )

        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, max_prompt_len - 1:]

        # --- Decode & compute rewards ---
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False,
        )
        completions = [[{"role": "assistant", "content": c}] for c in completions_text]

        rewards_per_func = torch.zeros(BG, len(self.reward_funcs), device=device)
        # Build reward kwargs: each prompt's fields repeated G times, concatenated
        reward_kwargs = {}
        for key in inputs[0].keys():
            if key != "prompt":
                reward_kwargs[key] = []
                for b in range(B):
                    reward_kwargs[key].extend([inputs[b][key]] * G)
        reward_prompts = []
        for b in range(B):
            reward_prompts.extend([prompts[b]] * G)

        for i, reward_func in enumerate(self.reward_funcs):
            output_rewards = reward_func(
                prompts=reward_prompts,
                completions=completions,
                **reward_kwargs,
            )
            rewards_per_func[:, i] = torch.tensor(
                output_rewards, dtype=torch.float32, device=device,
            )

        # --- Gather across GPUs & compute group-normalised advantages ---
        # After gather: [num_gpus * B * G, num_funcs].  Each consecutive block of G
        # rows belongs to one prompt → correct grouping for GRPO.
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        # Apply per-function weights (e.g. accuracy=1.0, format=0.2)
        rw = torch.tensor(self.reward_weights, device=rewards_per_func.device, dtype=rewards_per_func.dtype)
        rewards = (rewards_per_func * rw).sum(dim=1)

        mean_grouped = rewards.view(-1, G).mean(dim=1)
        std_grouped  = rewards.view(-1, G).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(G, dim=0)
        std_grouped  = std_grouped.repeat_interleave(G, dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * BG,
            (self.accelerator.process_index + 1) * BG,
        )
        advantages = advantages[process_slice]

        # --- Metrics ---
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        ).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            name = getattr(reward_func, "__name__", f"reward_{i}")
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped).mean().item()
        )

        # --- Prominent progress summary (rank 0 only) ---
        if self.accelerator.is_main_process:
            _step = self.state.global_step + 1
            _total = self.state.max_steps or "?"
            _acc = self._metrics["rewards/accuracy_reward"][-1] if "rewards/accuracy_reward" in self._metrics else 0
            _fmt = self._metrics["rewards/format_reward"][-1] if "rewards/format_reward" in self._metrics else 0
            _rew = self._metrics["reward"][-1]
            _clen = self._metrics["completion_length"][-1]
            print(
                f"\n{'='*70}\n"
                f"  [STEP {_step}/{_total}]  "
                f"acc_reward={_acc:.3f}  fmt_reward={_fmt:.3f}  "
                f"reward={_rew:.3f}  comp_len={_clen:.0f}\n"
                f"{'='*70}\n",
                flush=True,
            )

        # Free BG-expanded pixel values — no longer needed after ref/old logps.
        # Only pass B-sized multimodal inputs for pre-computing inputs_embeds.
        del multimodal_inputs
        torch.cuda.empty_cache()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "prompt_ids_B": prompt_ids_B,       # [B, max_P] — non-expanded
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "num_generations": G,
            **{f"mm_B_{k}": v for k, v in multimodal_inputs_B.items()},
        }

    # Maximum completion tokens to keep for the training forward pass.
    # Longer completions are truncated to cap memory.  We skip the model's lm_head
    # and apply it per-sample on completion-only hidden states with gradient
    # checkpointing, so peak logits memory is only ~0.6 GiB (1 sample) instead of
    # ~37 GiB (all 16 samples × full sequence).  Average completions are 400-500
    # tokens, so 1024 rarely truncates anything.
    _MAX_FORWARD_COMP = 1024

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """GRPO loss computation."""
        if return_outputs:
            raise ValueError("return_outputs is not supported")

        result = self._generate_and_score_completions(inputs, model)

        prompt_ids = result["prompt_ids"]
        prompt_mask = result["prompt_mask"]
        completion_ids = result["completion_ids"]
        completion_mask = result["completion_mask"]
        advantages = result["advantages"]
        old_per_token_logps = result["old_per_token_logps"]
        ref_per_token_logps = result["ref_per_token_logps"]

        # --- Truncate completions to cap memory for the training forward ---
        C = completion_ids.size(1)
        max_comp = self._MAX_FORWARD_COMP
        if C > max_comp:
            completion_ids = completion_ids[:, :max_comp]
            completion_mask = completion_mask[:, :max_comp]
            if old_per_token_logps is not None:
                old_per_token_logps = old_per_token_logps[:, :max_comp]
            if ref_per_token_logps is not None:
                ref_per_token_logps = ref_per_token_logps[:, :max_comp]

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_length = prompt_ids.size(1)  # = max_prompt_len (left-padded)

        # Full attention mask for forward pass (includes latent tokens)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx_c = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx_c[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx_c = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        full_completion_mask = (seq_idx_c <= eos_idx_c.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, full_completion_mask], dim=1)

        # Pre-compute inputs_embeds: run vision encoder only on B unique prompts (not B*G),
        # then expand to B*G.  This avoids 8x redundant vision encoding per prompt.
        prompt_ids_B = result["prompt_ids_B"]
        G = result["num_generations"]
        mm_B = {k.replace("mm_B_", ""): result[k] for k in result if k.startswith("mm_B_")}
        prompt_mask_B = prompt_mask[::G]  # take every G-th row to get B unique masks

        inputs_embeds_BG, rope_mm = self._precompute_inputs_embeds(
            model, prompt_ids_B, completion_ids, prompt_mask_B, mm_B, G
        )
        del mm_B, result  # free B-sized pixel values and remaining result tensors
        torch.cuda.empty_cache()

        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask,
            inputs_embeds=inputs_embeds_BG,
            prompt_length=prompt_length,
            **rope_mm,
        )
        del inputs_embeds_BG  # free after forward pass
        # per_token_logps is already completion-only (prompt logits discarded inside
        # _get_per_token_logps when prompt_length > 0)

        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # Policy gradient loss (completion_mask already excludes latent tokens)
        epsilon = getattr(self.args, "epsilon", 0.2)
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        pg_loss = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
        pg_loss = (pg_loss * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)

        # KL penalty
        if ref_per_token_logps is not None and self.beta != 0.0:
            kl = per_token_logps - ref_per_token_logps
            kl_loss = (kl * completion_mask).sum(1) / completion_mask.sum(1).clamp(min=1)
            loss = (pg_loss + self.beta * kl_loss).mean()
        else:
            loss = pg_loss.mean()

        self._metrics["loss"].append(loss.item())
        return loss

    def get_train_dataloader(self):
        """Dataloader without repetition — each prompt appears once per step.
        Generation of num_generations completions per prompt is batched
        inside _generate_and_score_completions (batch_size = B * G)."""
        from torch.utils.data import DataLoader, RandomSampler

        sampler = RandomSampler(
            self.train_dataset,
            generator=torch.Generator().manual_seed(self.args.data_seed),
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def log(self, logs, *args, **kwargs):
        """Merge custom metrics into logs, then write to TensorBoard."""
        if hasattr(self, "_metrics"):
            for key, values in self._metrics.items():
                if values:
                    logs[key] = sum(values) / len(values)
            self._metrics.clear()
        # Write to TensorBoard (only on main process)
        if hasattr(self, "_tb_writer") and self._tb_writer is not None:
            step = self.state.global_step
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)
            self._tb_writer.flush()
        super().log(logs, *args, **kwargs)


# ============================================================
# Main
# ============================================================

def main(script_args, training_args, model_args):
    rank = os.environ.get('LOCAL_RANK', '?')
    def _log(msg):
        print(f"[RANK {rank}] >>> {msg}", flush=True)

    _log("Entered main()")
    # -------------------------------------------------------
    # Directory layout (mirrors SFT train.sh / main.py):
    #   output_base/
    #     checkpoints/   <- model weights & optimizer states
    #     tensorboard/   <- TensorBoard event files
    #   rl/frame_cache/  <- extracted video frames (fixed location, reused across runs)
    # -------------------------------------------------------
    output_base = training_args.output_dir          # e.g. .../outputs/4dthinker_rl_grpo_3B
    ckpt_dir    = os.path.join(output_base, "checkpoints")
    tb_dir      = os.path.join(output_base, "tensorboard")
    # frame_cache lives under rl/ project root, not under output_dir, so it persists across experiments
    frame_cache_dir = os.path.join(pathlib.Path(__file__).resolve().parents[4], "frame_cache")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(frame_cache_dir, exist_ok=True)

    # Redirect HuggingFace output_dir → checkpoints sub-dir
    training_args.output_dir = ckpt_dir
    # Point TensorBoard logging to tensorboard sub-dir
    training_args.logging_dir = tb_dir
    training_args.report_to = ["tensorboard"]
    training_args.logging_strategy = "steps"

    _log("Loading dataset (frames should be pre-extracted) ...")
    dataset = load_4dthinker_dataset(script_args.data_file_paths, frame_cache_dir)
    _log(f"Dataset loaded. size={len(dataset)}")

    reward_funcs = [REWARD_FUNCS_REGISTRY[f] for f in script_args.reward_funcs]
    _log(f"Reward functions: {script_args.reward_funcs}, weights: {training_args.reward_weights}")

    _log("Initializing FourDThinkerGRPOTrainer (loading model + reference model) ...")
    trainer = FourDThinkerGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Attach a TensorBoard SummaryWriter on the main process
    trainer._tb_writer = None
    if trainer.accelerator.is_main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
            trainer._tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard logging → {tb_dir}")
        except ImportError:
            logger.warning("tensorboard not installed; skipping SummaryWriter")

    _log("Starting training ...")
    if list(pathlib.Path(ckpt_dir).glob("checkpoint-*")):
        _log("Resuming from checkpoint.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if trainer._tb_writer is not None:
        trainer._tb_writer.close()

    trainer.save_model(ckpt_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    import sys
    print(f"[RANK {os.environ.get('LOCAL_RANK', '?')}] >>> Script started, parsing args ...", flush=True)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    print(f"[RANK {os.environ.get('LOCAL_RANK', '?')}] >>> parse_args_and_config ...", flush=True)
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(f"[RANK {os.environ.get('LOCAL_RANK', '?')}] >>> Args parsed. Entering main() ...", flush=True)
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        logger.info("ZeRO-3 detected, applying qwen2_5vl forward monkey patch")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
