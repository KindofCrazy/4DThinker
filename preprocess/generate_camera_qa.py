#!/usr/bin/env python3
"""
Process video data: construct camera movement multiple-choice questions
with CoT reasoning based on static object masks and Gemini verification.

Usage:
    python generate_qa.py <video_id>
    python generate_qa.py  # uses default sample
"""

import os
import json
import sys
import random
import time
import base64
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from pathlib import Path
from openai import OpenAI

# ============ Configuration ============
BASE_DIR = os.environ.get("DATA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Gemini models: gemini-3-flash-preview highest priority, then gemini-3-pro-preview
AVAILABLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# All possible camera movements (full names, no abbreviations)
ALL_CAMERA_MOVEMENTS = [
    "Pan Right",
    "Pan Left",
    "Dolly In",
    "Dolly Out",
    "Roll Clockwise",
    "Roll Counter-Clockwise",
    "Tilt Up",
    "Tilt Down",
    "Pedestal Up",
    "Pedestal Down",
    "Truck Left",
    "Truck Right",
]

# Map from abbreviated names (in data) to full names
MOVEMENT_FULL_NAME = {
    "Pan Right": "Pan Right",
    "Pan Left": "Pan Left",
    "Dolly In": "Dolly In",
    "Dolly Out": "Dolly Out",
    "Roll CW": "Roll Clockwise",
    "Roll CCW": "Roll Counter-Clockwise",
    "Tilt Up": "Tilt Up",
    "Tilt Down": "Tilt Down",
    "Pedestal Up": "Pedestal Up",
    "Pedestal Down": "Pedestal Down",
    "Truck Left": "Truck Left",
    "Truck Right": "Truck Right",
}

# Opposite movements for distractor generation
OPPOSITES = {
    "Pan Right": "Pan Left", "Pan Left": "Pan Right",
    "Dolly In": "Dolly Out", "Dolly Out": "Dolly In",
    "Roll Clockwise": "Roll Counter-Clockwise", "Roll Counter-Clockwise": "Roll Clockwise",
    "Tilt Up": "Tilt Down", "Tilt Down": "Tilt Up",
    "Pedestal Up": "Pedestal Down", "Pedestal Down": "Pedestal Up",
    "Truck Left": "Truck Right", "Truck Right": "Truck Left",
}


# ============ Gemini API Utilities ============

def encode_image(image_path: str) -> tuple:
    """Read image and return base64 encoding and MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return image_data, mime_type


def call_gemini(text: str, image_paths=None, max_retries=5):
    """
    Call Gemini API with model rotation and retry logic.
    Priority: gemini-3-flash-preview > gemini-3-pro-preview > others.
    On failure, rotate to next model and retry.
    """
    content = []

    # Add images first
    if image_paths:
        for img_path in image_paths:
            image_data, mime_type = encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
            })

    # Add text
    content.append({"type": "text", "text": text})

    message = {"role": "user", "content": content}

    for attempt in range(max_retries):
        model = AVAILABLE_MODELS[attempt % len(AVAILABLE_MODELS)]
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[message],
                stream=False,
                max_tokens=65536,
                extra_headers={
                    "M-TraceId": f"4dthinker_{int(time.time())}_{random.randint(0, 9999)}"
                }
            )
            response_text = result.choices[0].message.content
            print(f"    [Gemini] Model={model} -> OK (len={len(response_text)})")
            return response_text
        except Exception as e:
            print(f"    [Retry {attempt+1}/{max_retries}] Model={model} failed: {e}")
            time.sleep(2)

    return None


# ============ Mask Overlay (RED color) ============

def create_masked_overlay(frame_path, mask_path, output_path, color=(0, 0, 255), alpha=0.4):
    """
    Overlay static mask on frame image with RED highlight and save.
    color is BGR: (0, 0, 255) = red in OpenCV.
    Returns True on success.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"    WARNING: Cannot read frame: {frame_path}")
        return False

    mask = np.load(mask_path)
    h, w = frame.shape[:2]

    # Resize mask if needed
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h)) > 127
    else:
        mask = mask > 127

    mask = mask.astype(bool)

    # Check mask has meaningful content (at least 0.1% of image)
    mask_ratio = mask.sum() / (h * w)
    if mask_ratio < 0.001:
        print(f"    WARNING: Mask too small ({mask_ratio:.4%}) in {mask_path}")
        return False

    # Create overlay with red highlight
    overlay = frame.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = np.clip(
        frame[mask].astype(np.float32) * (1 - alpha) + color_arr * alpha,
        0, 255
    ).astype(np.uint8)

    # Draw red contour for clearer boundary
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    return True


# ============ Gemini Verification (detailed prompt) ============

def check_same_object_gemini(masked_images: list, static_object_name: str) -> bool:
    """
    Use Gemini to verify the masked region in multiple frames shows the same
    static object with consistent mask coverage.
    """
    prompt = '''You are given {num_frames} frames extracted from a video sequence. \
In each frame, a region has been highlighted with a red semi-transparent overlay \
and a red contour outline. The intended target object is '{obj}'.

Please carefully evaluate ALL of the following criteria:
1. **Object Identity**: Does the red-highlighted region in EVERY frame correspond \
to the same '{obj}' object? (Not a different object or background.)
2. **Mask Consistency**: Is the mask region reasonably consistent across frames? \
The highlighted area should cover roughly the same portion of the object in each frame, \
without dramatically shifting to a completely different part of the scene.
3. **Mask Quality**: Does the mask reasonably cover the target object (not just a tiny \
fragment or an excessively large area that includes unrelated background)?
4. **Object Visibility**: Is the '{obj}' clearly visible and identifiable in all frames?

If ALL criteria are satisfied, answer 'YES'. If ANY criterion fails, answer 'NO'.
Answer with ONLY 'YES' or 'NO', nothing else.'''.format(
        num_frames=len(masked_images),
        obj=static_object_name,
    )

    response = call_gemini(prompt, masked_images)
    if response is None:
        print("    WARNING: Gemini verification failed (no response)")
        return False

    answer = response.strip().upper()
    result = "YES" in answer
    print(f"    Gemini verification: '{response.strip()}' -> {result}")
    return result


# ============ Gemini CoT Generation ============

def validate_cot_format(response: str, expected_image_count: int) -> bool:
    """
    Validate that the CoT response follows the required format:
    - <thinking> ... </thinking> with substantial reasoning and <output_image> placeholders
    - <answer> ... </answer> after </thinking>
    - <thinking> must NOT contain <answer> tags
    - <output_image> tags must appear exactly expected_image_count times
    - <output_image> tags should be grouped together (all in a short span), not scattered
    """
    response = response.strip()
    if "<thinking>" not in response or "</thinking>" not in response:
        return False
    if "<answer>" not in response or "</answer>" not in response:
        return False

    # Extract think content
    think_start = response.index("<thinking>") + len("<thinking>")
    think_end = response.index("</thinking>")
    think_content = response[think_start:think_end].strip()

    # think must have meaningful content (at least 100 chars)
    if len(think_content) < 100:
        return False

    # think must contain <output_image> placeholders
    if "<output_image>" not in think_content:
        return False

    # Check exact count of <output_image>
    tag_count = think_content.count("<output_image>")
    if tag_count != expected_image_count:
        return False

    # think must NOT contain <answer> tags
    if "<answer>" in think_content:
        return False

    # <answer> must come after </thinking>
    answer_pos = response.index("<answer>")
    think_end_pos = response.index("</thinking>")
    if answer_pos < think_end_pos:
        return False

    # Check that <output_image> tags are grouped (all within a ~300 char span)
    first_tag_pos = think_content.index("<output_image>")
    last_tag_pos = think_content.rindex("<output_image>")
    tag_span = last_tag_pos - first_tag_pos
    # Allow ~150 chars per tag for the connecting text between them
    max_span = expected_image_count * 150
    if tag_span > max_span:
        return False

    return True


def generate_cot_with_gemini(
    all_video_frames: list,
    masked_images: list,
    key_frames_indices: list,
    static_object_name: str,
    correct_movements_full: list,
    correct_label: str,
    options: dict,
    start_frame: int,
    end_frame: int,
    total_frames: int,
    question_text: str,
) -> str:
    """
    Call Gemini to generate a detailed CoT reasoning for camera movement.

    Input images to Gemini (in order):
      1. Video Frames: ALL frames of the entire video (0..total_frames-1)
      2. Masked Frames: key frames with static object highlighted in red
         — "mental imagery", referenced via <output_image> (count may differ)

    The CoT must first temporally locate the relevant segment within the
    full video, then analyse camera motion using the masked frames.

    Returns the CoT string, or None if all retries fail (caller should skip).
    """
    max_attempts = 10
    num_video = len(all_video_frames)
    num_masked = len(masked_images)

    correct_answer_text = format_movement(correct_movements_full)
    masked_frame_desc = ", ".join([f"{idx}s" for idx in key_frames_indices])

    # ---- Build full instruction text (merged into user message, no system role) ----
    # NOTE: This Gemini API endpoint does not reliably support the "system" role.
    # All instructions MUST go into the "user" message to be seen by the model.
    instruction_text = '''\
You are an advanced multimodal reasoning assistant. Your task is to generate \
a comprehensive, step-by-step Chain of Thought (CoT) reasoning process that \
leads to the correct option for a given multiple-choice question.

**Input Data Provided to You:**
1.  **Video Frames:** The complete video as a sequence of {num_video} frames \
(from 0s to {total_frames}s). \
2.  **Masked Frames:** {num_masked} key frames (at {masked_frame_desc}) where \
the static object '{obj}' is highlighted in red overlay and red contour. These \
correspond to specific moments within the segment {start_frame}s–{end_frame}s.
3.  **Multiple Choice Question:**
{question_text}
4.  **Correct Answer:** {correct_label}. {correct_answer_text}

**Your Goal:**
You must engineer the reasoning process. You need to output a thought \
process that logically uses the visual inputs to derive the provided Correct Answer.

**Critical Rules for `<thinking>` and `<output_image>`:**
1.  **Mental Imagery as Output (NOT Input):** The Masked Frames are NOT part of \
your input — they represent the **output of your own mental imagination**. During \
reasoning, you should describe a process where you *mentally visualize* highlighting \
the static object '{obj}' in your mind, imagining where it appears across frames. \
Each `<output_image>` tag represents one such mental image you produce in your mind's \
eye, where you picture the object highlighted (as shown in the Masked Frames). \
Think of it as: "In my mind, I picture the '{obj}' highlighted across the key moments, \
forming a progression from <output_image> to <output_image> to <output_image>."
2.  **Strict Usage of `<output_image>`:**
    *   Each `<output_image>` is a **mental image you generate** — your imagination \
of the static object '{obj}' highlighted in red at a specific moment in time. \
You **must** use the placeholder `<output_image>` to represent these imagined frames.
    *   **Quantity & Order:** You must use exactly {num_masked} `<output_image>` tags \
(one per imagined frame), strictly in chronological order.
    *   **Grouping:** You must present these tags **all together in one sentence** \
to show the mental progression.
        *   *Correct Format:* "...I mentally visualize the change from <output_image> to \
<output_image> to <output_image>, where the '{obj}' is highlighted in each imagined frame."
3.  **Negative Constraint (Important):**
    *   **Do NOT** use the `<output_image>` tag again in your subsequent analysis text.
    *   When analyzing the shift (e.g., describing position changes), refer to them \
using natural language like "In the first frame," "In the final view," or "In this \
sequence," without re-triggering the tag.
4.  **Reasoning Flow:**
    *   **Temporal Localization:** The video has {num_video} frames covering \
0s–{total_frames}s. You must first identify and narrow down to the relevant time \
segment ({start_frame}s–{end_frame}s) that the question asks about. Describe what \
you observe in the overall video and how you locate the target interval.
    *   **Scene Context:** Briefly describe what happens in the target \
segment ({start_frame}s–{end_frame}s).
    *   **Strategy:** State the need for a static reference point and \
introduce the chosen object.
    *   **Visualization:** Describe your mental process of imagining the \
object highlighted, and present the `<output_image>` sequence as your own \
mental imagery output (strictly matching count of {num_masked}).
    *   **Analysis:** Describe the pixel displacement observed in that \
sequence (using text only, no tags).
    *   **Conclusion:** Map the physical movement to the correct option.

**Output Format:**
<thinking>
[Detailed reasoning ...]
</thinking>
<answer>
[The Correct Option, e.g., "A. Pan Right"]
</answer>

**Example:**
(Suppose the video has 60 frames covering 0s–60s, and the question asks about 20s–30s.)
<thinking>
The question asks about camera movement between 20s and 30s, so I first need to locate this interval. Looking \
through the full video sequence, the frames around the 20s–30s mark show a busy \
street scene with cars and pedestrians.

Now focusing on this segment, I see cars moving forward and pedestrians walking. \
The dynamic nature of the foreground makes it hard to judge the camera motion \
directly from the moving traffic.

To solve this, I need to isolate a static object in the background to serve as a \
reference point. I focus my attention on the distant building. \

In my mind, I imagine highlighting this building across the key moments to track \
its apparent motion. I mentally picture the building with a red highlight, forming \
a clear visual progression in my imagination: from <output_image> to <output_image> \
to <output_image>.

Analyzing this imagined sequence, I observe a distinct horizontal shift. In \
the first mental image, the highlighted building is located near the right edge of the \
frame. By the final imagined frame, the building has drifted \
significantly toward the left side.

Since a stationary object appears to move from Right to Left, the camera must be \
physically panning in the opposite direction. This confirms the camera is moving \
to the right.

Comparing this conclusion with the given options, it matches Option B.
</thinking>
<answer>B. Pan Right</answer>

The images follow below (first {num_video} are Video Frames of the complete video, \
next {num_masked} are Masked Frames for the target segment).'''.format(
        num_video=num_video,
        num_masked=num_masked,
        total_frames=total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        masked_frame_desc=masked_frame_desc,
        obj=static_object_name,
        question_text=question_text,
        correct_label=correct_label,
        correct_answer_text=correct_answer_text,
    )

    # Retry up to max_attempts times, rotating through AVAILABLE_MODELS by priority
    for attempt in range(max_attempts):
        model = AVAILABLE_MODELS[attempt % len(AVAILABLE_MODELS)]
        print(f"    CoT attempt {attempt + 1}/{max_attempts} (model={model})...")

        try:
            # Build single user message: instructions -> all video frames -> masked images
            user_content = [{"type": "text", "text": instruction_text}]

            for img_path in all_video_frames:
                image_data, mime_type = encode_image(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                })

            # Label the masked frames
            user_content.append({
                "type": "text",
                "text": "--- The following {n} images are Masked Frames (your mental imagery, use <output_image> for each): ---".format(n=num_masked)
            })

            for img_path in masked_images:
                image_data, mime_type = encode_image(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                })

            result = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_content},
                ],
                stream=False,
                max_tokens=8192,
                extra_headers={
                    "M-TraceId": f"4dthinker_cot_{int(time.time())}_{random.randint(0, 9999)}"
                },
            )
            response = result.choices[0].message.content.strip()
            print(f"    [Gemini] Model={model} -> OK (len={len(response)})")
        except Exception as e:
            print(f"    [Retry {attempt + 1}/{max_attempts}] Model={model} failed: {e}")
            time.sleep(2)
            continue

        if validate_cot_format(response, num_masked):
            print(f"    CoT format validated OK")
            return response

        tag_count = response.count("<output_image>")
        print(f"    WARNING: CoT format invalid "
              f"(found {tag_count} <output_image>, expected {num_masked}), "
              f"preview: {response[:200]}...")
        time.sleep(1)

    # All 10 attempts failed — return None so caller skips this segment
    print(f"    FAILED: All {max_attempts} CoT attempts exhausted, skipping this segment")
    return None


# ============ Gemini Question Phrasing ============

def generate_question_with_gemini(
    start_frame: int,
    end_frame: int,
    options: dict,
    segment_index: int,
) -> str:
    """
    Use Gemini to generate a diverse, natural question phrasing for the
    camera movement multiple-choice question. Each call produces a different style.
    """
    options_str = "\n".join([f"{label}. {text}" for label, text in options.items()])

    # Provide several style hints to encourage diversity
    style_hints = [
        "a concise and direct style",
        "a descriptive style that sets the scene",
        "a style that emphasizes observation and analysis",
        "a conversational and engaging tone",
        "a formal and technical cinematography style",
        "a style that asks the viewer to identify the technique",
    ]
    chosen_style = style_hints[segment_index % len(style_hints)]

    prompt = '''\
You are writing multiple-choice questions about camera movements in videos.

Context: A video is given as input. We want to ask about the camera movement \
during the time segment from {start_frame}s to {end_frame}s.

The answer options are:
{options_str}

Please write a single question (in English) that asks the viewer to identify \
the camera movement during this time segment. Use {chosen_style}.

Rules:
- The question should naturally reference the time range ({start_frame}s to {end_frame}s).
- Do NOT mention any specific objects, masks, or highlights.
- Do NOT include the options in your output — just the question text itself.
- Keep it to 1-2 sentences.
- Output ONLY the question text, nothing else.'''.format(
        start_frame=start_frame,
        end_frame=end_frame,
        options_str=options_str,
        chosen_style=chosen_style,
    )

    response = call_gemini(prompt)
    if response and len(response.strip()) > 10:
        question_body = response.strip().strip('"').strip("'")
        # Append options
        return f"{question_body}\n\n{options_str}"

    # Fallback: simple template
    return f"What is the camera movement from {start_frame}s to {end_frame}s?\n\n{options_str}"


# ============ Question & Distractor Construction ============

def normalize_movement(movement: str) -> str:
    """Convert abbreviated movement name to full name."""
    return MOVEMENT_FULL_NAME.get(movement, movement)


def format_movement(movements: list) -> str:
    """Format a list of movements into a readable string."""
    if len(movements) == 1:
        return movements[0]
    return ", ".join(movements[:-1]) + " and " + movements[-1]


def generate_distractors(correct_movements: list, num_distractors: int = 3) -> list:
    """
    Generate plausible wrong camera movement options.
    All movements use full names (no abbreviations).
    """
    correct_set = set(correct_movements)
    distractors = []
    used_sets = [frozenset(correct_set)]

    # Strategy 1: Opposite of the correct answer
    opposite_movements = []
    for m in correct_movements:
        if m in OPPOSITES:
            opposite_movements.append(OPPOSITES[m])
        else:
            opposite_movements.append(m)  # keep if no opposite defined
    opp_set = frozenset(opposite_movements)
    if opp_set != frozenset(correct_set) and opp_set not in used_sets:
        distractors.append(opposite_movements)
        used_sets.append(opp_set)

    # Strategy 2: If multi-movement, create partial overlap distractors
    if len(correct_movements) > 1:
        for i, m in enumerate(correct_movements):
            if len(distractors) >= num_distractors:
                break
            alt = list(correct_movements)
            if m in OPPOSITES:
                alt[i] = OPPOSITES[m]
            else:
                others = [x for x in ALL_CAMERA_MOVEMENTS if x not in correct_set]
                if others:
                    alt[i] = random.choice(others)
            alt_fs = frozenset(alt)
            if alt_fs not in used_sets:
                distractors.append(alt)
                used_sets.append(alt_fs)

    # Strategy 3: Random movements to fill remaining slots
    available = [m for m in ALL_CAMERA_MOVEMENTS if m not in correct_set]
    random.shuffle(available)

    while len(distractors) < num_distractors and available:
        if len(correct_movements) > 1 and random.random() < 0.4 and len(available) >= len(correct_movements):
            n = len(correct_movements)
            combo = available[:n]
            available = available[n:]
            combo_fs = frozenset(combo)
            if combo_fs not in used_sets:
                distractors.append(combo)
                used_sets.append(combo_fs)
        else:
            single = [available.pop(0)]
            single_fs = frozenset(single)
            if single_fs not in used_sets:
                distractors.append(single)
                used_sets.append(single_fs)

    return distractors[:num_distractors]


# ============ Main Processing ============

def get_all_frame_paths(frames_dir: str) -> list:
    """Get sorted list of ALL frame image paths in the frames directory."""
    if not os.path.isdir(frames_dir):
        return []
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    return [os.path.join(frames_dir, f) for f in frame_files]


def process_single_data(video_id: str):
    """
    Process all instructions for a single video data entry.
    Returns a list of QA result dicts.
    """
    data_dir = os.path.join(DATA_DIR, video_id)
    data_file = os.path.join(data_dir, "data.jsonl")

    if not os.path.exists(data_file):
        print(f"ERROR: data.jsonl not found at {data_file}")
        return []

    # Load data
    with open(data_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            print(f"ERROR: data.jsonl is empty")
            return []
        data = json.loads(line)

    instructions = data.get("instructions", {})
    static_object = data.get("static_object", "")
    frames_dir = os.path.join(data_dir, "frames")
    masks_static_dir = os.path.join(data_dir, "masks_static")

    if not static_object or static_object == "null":
        print(f"ERROR: No static_object defined for {video_id}")
        return []

    if not os.path.isdir(masks_static_dir):
        print(f"ERROR: masks_static directory not found for {video_id}")
        return []

    # Get ALL original frame paths (for the entire video)
    all_original_frame_paths = get_all_frame_paths(frames_dir)
    total_frames = len(all_original_frame_paths)
    print(f"Total frames in video: {total_frames}")

    # Create output directory for masked overlay images
    overlay_dir = os.path.join(data_dir, "masked_static_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    results = []

    for segment_key, movements_raw in instructions.items():
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_key} -> {movements_raw}")
        print(f"{'='*60}")

        # Normalize movement names to full form
        movements = [normalize_movement(m) for m in movements_raw]

        # Parse frame range
        parts = segment_key.split("->")
        start_frame = int(parts[0])
        end_frame = int(parts[1])

        # ---- Step 1: Check masks_static existence for start and end frames ----
        start_mask = os.path.join(masks_static_dir, f"frame_{start_frame:04d}.npy")
        end_mask = os.path.join(masks_static_dir, f"frame_{end_frame:04d}.npy")

        if not os.path.exists(start_mask):
            print(f"  SKIP: No masks_static for start frame {start_frame}")
            continue
        if not os.path.exists(end_mask):
            print(f"  SKIP: No masks_static for end frame {end_frame}")
            continue

        start_frame_path = os.path.join(frames_dir, f"frame_{start_frame:04d}.jpg")
        end_frame_path = os.path.join(frames_dir, f"frame_{end_frame:04d}.jpg")

        if not os.path.exists(start_frame_path) or not os.path.exists(end_frame_path):
            print(f"  SKIP: Missing frame images")
            continue

        # ---- Step 2: Create masked overlay images (RED) ----
        seg_tag = segment_key.replace("->", "_to_")
        key_frames_info = []  # (frame_idx, overlay_path, original_frame_path)

        start_overlay = os.path.join(overlay_dir, f"seg_{seg_tag}_f{start_frame:04d}.jpg")
        end_overlay = os.path.join(overlay_dir, f"seg_{seg_tag}_f{end_frame:04d}.jpg")

        if not create_masked_overlay(start_frame_path, start_mask, start_overlay):
            print(f"  SKIP: Failed to create overlay for start frame {start_frame}")
            continue
        if not create_masked_overlay(end_frame_path, end_mask, end_overlay):
            print(f"  SKIP: Failed to create overlay for end frame {end_frame}")
            continue

        key_frames_info.append((start_frame, start_overlay, start_frame_path))

        # ---- Step 3: For multi-movement segments, find intermediate frames ----
        # Number of intermediate frames = number of transitions (arrows),
        # i.e. len(movements) - 1, capped at 1-2.
        if len(movements) > 1:
            mid_range = list(range(start_frame + 1, end_frame))
            available_mid = [
                f for f in mid_range
                if os.path.exists(os.path.join(masks_static_dir, f"frame_{f:04d}.npy"))
            ]

            num_intermediate = min(len(movements) - 1, 2)  # 1-2 based on arrow count
            num_sample = min(num_intermediate, len(available_mid))
            if num_sample > 0:
                sampled_mids = sorted(random.sample(available_mid, num_sample))
                for mid_f in sampled_mids:
                    mid_frame_path = os.path.join(frames_dir, f"frame_{mid_f:04d}.jpg")
                    mid_mask_path = os.path.join(masks_static_dir, f"frame_{mid_f:04d}.npy")
                    mid_overlay = os.path.join(overlay_dir, f"seg_{seg_tag}_f{mid_f:04d}.jpg")

                    if os.path.exists(mid_frame_path):
                        if create_masked_overlay(mid_frame_path, mid_mask_path, mid_overlay):
                            key_frames_info.append((mid_f, mid_overlay, mid_frame_path))
                            print(f"  Added intermediate frame {mid_f}")
            else:
                print(f"  NOTE: No intermediate frames with masks_static found")

        # Add end frame last
        key_frames_info.append((end_frame, end_overlay, end_frame_path))

        # Sort by frame index
        key_frames_info.sort(key=lambda x: x[0])
        key_frame_indices = [info[0] for info in key_frames_info]
        print(f"  Key frames: {key_frame_indices}")

        # ---- Step 4: Verify same object with Gemini (detailed criteria) ----
        all_overlay_paths = [info[1] for info in key_frames_info]
        print(f"  Verifying {len(all_overlay_paths)} frames show same object '{static_object}'...")

        same_object = check_same_object_gemini(all_overlay_paths, static_object)
        if not same_object:
            print(f"  SKIP: Gemini says masks are inconsistent for '{static_object}'")
            continue

        print(f"  PASS: Same object verified!")

        # ---- Step 5: Use ALL video frames (not just the segment) ----
        print(f"  Video frames: {total_frames} (full video), "
              f"segment: {start_frame}->{end_frame}, "
              f"masked key frames: {len(all_overlay_paths)}")

        # ---- Step 6: Construct multiple-choice options ----
        correct_answer = movements
        distractors = generate_distractors(correct_answer, num_distractors=3)

        all_options = [correct_answer] + distractors
        random.shuffle(all_options)
        correct_idx = all_options.index(correct_answer)

        option_labels = ["A", "B", "C", "D"]
        options = {}
        for i, opt in enumerate(all_options):
            options[option_labels[i]] = format_movement(opt)

        correct_label = option_labels[correct_idx]

        # ---- Step 7: Generate diverse question via Gemini ----
        segment_index = len(results)  # use index for style variation
        print(f"  Generating question phrasing with Gemini...")
        question_text = generate_question_with_gemini(
            start_frame, end_frame, options, segment_index
        )

        # ---- Step 8: Generate CoT with Gemini ----
        # Input: ALL video frames (full video) + masked key frames
        print(f"  Generating CoT with Gemini "
              f"({total_frames} video frames + {len(all_overlay_paths)} masked)...")
        cot = generate_cot_with_gemini(
            all_video_frames=all_original_frame_paths,
            masked_images=all_overlay_paths,
            key_frames_indices=key_frame_indices,
            static_object_name=static_object,
            correct_movements_full=correct_answer,
            correct_label=correct_label,
            options=options,
            start_frame=start_frame,
            end_frame=end_frame,
            total_frames=total_frames,
            question_text=question_text,
        )

        if cot is None:
            print(f"  SKIP: CoT generation failed after all retries for segment {segment_key}")
            continue

        # ---- Step 9: Build result ----
        result = {
            "id": f"{video_id}_{segment_key}",
            "question": question_text.strip(),
            "options": options,
            "correct_answer": f"{correct_label}. {format_movement(correct_answer)}",
            "cot": cot,
            "mask_image_paths": all_overlay_paths,
            "original_image_paths": all_original_frame_paths,
        }

        results.append(result)
        print(f"  SUCCESS: Generated QA for segment {segment_key}")
        print(f"  Correct answer: {correct_label}. {format_movement(correct_answer)}")

    return results


def get_all_video_ids() -> list:
    """Scan DATA_DIR and return all video_id subdirectories that contain data.jsonl."""
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: DATA_DIR not found: {DATA_DIR}")
        return []
    video_ids = []
    for name in sorted(os.listdir(DATA_DIR)):
        sub_dir = os.path.join(DATA_DIR, name)
        if os.path.isdir(sub_dir) and os.path.exists(os.path.join(sub_dir, "data.jsonl")):
            video_ids.append(name)
    return video_ids


def load_completed_ids(output_file: str) -> set:
    """Load already-processed video_id prefixes from existing JSONL for resume."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # id format: "{video_id}_{segment_key}", extract video_id
                    item_id = obj.get("id", "")
                    # video_id is everything before the last underscore-separated segment key
                    # segment_key looks like "0->3", so id looks like "xxxx-xxx_0->3"
                    # Split on last occurrence of "_" followed by digits
                    vid = item_id.rsplit("_", 1)[0] if "_" in item_id else item_id
                    completed.add(vid)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"WARNING: Error reading existing output file: {e}")
    return completed


# Thread lock for writing to JSONL file
_write_lock = threading.Lock()


def process_and_write(video_id: str, output_file: str, progress_info: dict):
    """
    Process a single video and append results to the JSONL file (thread-safe).
    progress_info is a shared dict with 'done', 'success', 'total' counters and a lock.
    """
    try:
        results = process_single_data(video_id)

        if results:
            with _write_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with progress_info["lock"]:
                progress_info["success"] += 1
                progress_info["qa_count"] += len(results)

        with progress_info["lock"]:
            progress_info["done"] += 1
            done = progress_info["done"]
            total = progress_info["total"]
            success = progress_info["success"]
            qa_count = progress_info["qa_count"]

        num_results = len(results) if results else 0
        print(f"\n[Progress {done}/{total}] video={video_id} "
              f"-> {num_results} QA items  (total success: {success}, total QA: {qa_count})")

        return video_id, len(results) if results else 0

    except Exception as e:
        with progress_info["lock"]:
            progress_info["done"] += 1
            done = progress_info["done"]
            total = progress_info["total"]
        print(f"\n[Progress {done}/{total}] video={video_id} -> ERROR: {e}")
        return video_id, 0


def main():
    # ---- Configuration ----
    MAX_WORKERS = 16  # Number of concurrent threads
    output_file = os.path.join(BASE_DIR, "camera_data_qa_all.jsonl")

    # Support optional single video_id mode via command line
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        # Single video mode (backward compatible)
        video_id = sys.argv[1]
        print(f"{'#'*60}")
        print(f"Processing single video: {video_id}")
        print(f"{'#'*60}")
        results = process_single_data(video_id)
        if not results:
            print("\nNo QA items generated. Check the logs above for skip reasons.")
            return
        with _write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults appended to: {output_file}")
        print(f"Generated {len(results)} QA items")
        return

    # ---- Batch mode: process ALL video directories ----
    print(f"{'#'*60}")
    print(f"Scanning all video directories in {DATA_DIR}...")
    print(f"{'#'*60}")

    all_video_ids = get_all_video_ids()
    print(f"Found {len(all_video_ids)} video directories with data.jsonl")

    # Load already-completed video IDs for resume
    completed_ids = load_completed_ids(output_file)
    if completed_ids:
        print(f"Found {len(completed_ids)} already-processed videos, skipping them.")

    pending_ids = [vid for vid in all_video_ids if vid not in completed_ids]
    print(f"Pending: {len(pending_ids)} videos to process")

    if not pending_ids:
        print("All videos already processed. Nothing to do.")
        return

    # ---- Multi-threaded processing ----
    progress_info = {
        "done": 0,
        "success": 0,
        "qa_count": 0,
        "total": len(pending_ids),
        "lock": threading.Lock(),
    }

    print(f"\nStarting processing with {MAX_WORKERS} threads...")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_and_write, vid, output_file, progress_info): vid
            for vid in pending_ids
        }

        for future in as_completed(futures):
            video_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[FATAL] Unhandled exception for {video_id}: {e}")

    elapsed = time.time() - start_time

    print(f"\n{'#'*60}")
    print(f"ALL DONE!")
    print(f"Total videos processed: {progress_info['done']}/{progress_info['total']}")
    print(f"Successful videos: {progress_info['success']}")
    print(f"Total QA items generated: {progress_info['qa_count']}")
    print(f"Output file: {output_file}")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
