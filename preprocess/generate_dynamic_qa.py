#!/usr/bin/env python3
"""
Process video data: construct dynamic object movement multiple-choice questions
with CoT reasoning based on dynamic object masks and Gemini verification.

Usage:
    python generate_dynamic_qa.py <video_id>
    python generate_dynamic_qa.py           # batch mode: process all videos
"""

import os
import json
import sys
import math
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

# All possible object movement directions
ALL_MOVEMENT_DIRECTIONS = [
    "Moving Left",
    "Moving Right",
    "Moving Up",
    "Moving Down",
    "Moving Left-Up",
    "Moving Left-Down",
    "Moving Right-Up",
    "Moving Right-Down",
    "Moving Toward Camera",
    "Moving Away from Camera",
    "Stationary",
]

# Opposite directions for distractor generation
OPPOSITES = {
    "Moving Left": "Moving Right",
    "Moving Right": "Moving Left",
    "Moving Up": "Moving Down",
    "Moving Down": "Moving Up",
    "Moving Left-Up": "Moving Right-Down",
    "Moving Left-Down": "Moving Right-Up",
    "Moving Right-Up": "Moving Left-Down",
    "Moving Right-Down": "Moving Left-Up",
    "Moving Toward Camera": "Moving Away from Camera",
    "Moving Away from Camera": "Moving Toward Camera",
}


# ============ Bbox Conversion (from process_bbox.ipynb) ============

def smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that:
    1. Both dimensions are divisible by 'factor'.
    2. Total pixels within [min_pixels, max_pixels].
    3. Aspect ratio is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(
    bbox, orig_height, orig_width, factor=28,
    min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280,
):
    """Convert bbox coordinates to Qwen2.5-VL resized coordinate space."""
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))

    return [x1_new, y1_new, x2_new, y2_new]


def bbox_from_mask(mask: np.ndarray) -> list:
    """Extract bounding box [x1, y1, x2, y2] from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


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
                    "M-TraceId": f"4dthinker_dyn_{int(time.time())}_{random.randint(0, 9999)}"
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
    Overlay dynamic mask on frame image with RED highlight and save.
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

    # Draw contour for clearer boundary
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    return True


# ============ Gemini: Verify Same Dynamic Object ============

def check_dynamic_object_gemini(masked_images: list, dynamic_object_name: str) -> dict:
    """
    Use Gemini to verify which masked frames show the same dynamic (moving) object.
    Returns a dict with:
      - 'valid_indices': list of indices (0-based, within masked_images) that show the same moving object
      - 'object_name': Gemini's description of the identified object (or the provided name)
    """
    prompt = '''You are given {num_frames} frames extracted from a video sequence. \
In each frame, a region has been highlighted with a red semi-transparent overlay \
and a red contour outline. The intended target is a dynamic (moving) object, \
possibly described as '{obj}'.

Please carefully evaluate each frame:
1. **Is it a dynamic object?** The highlighted object must be something that is \
actually moving/in motion in the video (e.g., a person walking, a car driving, \
an animal running). If the highlighted region is a static/stationary object \
(like a building, tree, or fixed furniture), mark that frame as INVALID.
2. **Same object identity**: Among the frames with valid dynamic objects, do they \
all show the SAME moving object (same identity, not just same category)?
3. **Mask quality**: Does the mask reasonably cover the target object?

Output a JSON object with exactly this format (no other text):
{{
  "is_any_dynamic": true/false,
  "valid_frame_indices": [list of 0-based indices that show the same moving object],
  "object_description": "brief description of the identified moving object"
}}

If no frames show a dynamic (moving) object, set "is_any_dynamic" to false and \
"valid_frame_indices" to an empty list.
If some frames show the same moving object but others don't, include only the \
consistent ones in "valid_frame_indices".'''.format(
        num_frames=len(masked_images),
        obj=dynamic_object_name,
    )

    response = call_gemini(prompt, masked_images)
    if response is None:
        print("    WARNING: Gemini dynamic object verification failed (no response)")
        return {"valid_indices": [], "object_name": dynamic_object_name}

    # Parse JSON from response
    try:
        # Try to extract JSON from response (might have markdown fences)
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        is_dynamic = result.get("is_any_dynamic", False)
        valid_indices = result.get("valid_frame_indices", [])
        obj_desc = result.get("object_description", dynamic_object_name)

        print(f"    Gemini dynamic check: is_dynamic={is_dynamic}, "
              f"valid_indices={valid_indices}, obj='{obj_desc}'")

        if not is_dynamic:
            return {"valid_indices": [], "object_name": obj_desc}

        return {"valid_indices": valid_indices, "object_name": obj_desc}

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"    WARNING: Failed to parse Gemini response: {e}")
        print(f"    Response was: {response[:300]}")
        return {"valid_indices": [], "object_name": dynamic_object_name}


# ============ Gemini: Determine Movement Direction ============

def determine_movement_direction_gemini(
    all_video_frames: list,
    masked_images: list,
    key_frame_indices: list,
    dynamic_object_name: str,
    scene_summary: str,
    instructions: dict,
) -> str:
    """
    Use Gemini to determine the movement direction of the dynamic object.
    Returns the movement direction string (one of ALL_MOVEMENT_DIRECTIONS), or None.
    """
    directions_list = "\n".join([f"- {d}" for d in ALL_MOVEMENT_DIRECTIONS])
    instructions_str = json.dumps(instructions, ensure_ascii=False)
    frame_desc = ", ".join([f"{idx}s" for idx in key_frame_indices])

    prompt = '''\
You are given a video as a sequence of {num_video} frames (at 1 fps, so frame N = Ns).
You are also given {num_masked} key frames (at times: {frame_desc}) where a dynamic \
(moving) object '{obj}' is highlighted with a red overlay and contour.

**Scene context:** {scene_summary}
**Camera motion instructions (for reference, NOT the object movement):** {instructions}

Your task: Analyze the position change of the highlighted dynamic object '{obj}' \
across the masked key frames and determine its PRIMARY movement direction \
**as observed from the camera's viewpoint** (i.e., how the object moves within the \
camera frame).

IMPORTANT:
- All directions are defined **from the camera's perspective**: "Moving Left" means \
the object moves toward the left side of the camera frame, "Moving Up" means toward \
the top of the frame, etc.
- "Moving Toward Camera" means the object is getting closer to the camera (appears \
larger over time). "Moving Away from Camera" means getting farther (appears smaller).
- The camera may also be moving, so you need to separate the camera's own motion \
from the object's actual motion in the 3D scene. Focus on how the object's position \
changes **relative to the scene/world**, then express that as seen from the camera.
- Choose the single best direction that describes the object's dominant movement.

Available directions:
{directions_list}

Output ONLY the direction name from the list above, nothing else.'''.format(
        num_video=len(all_video_frames),
        num_masked=len(masked_images),
        frame_desc=frame_desc,
        obj=dynamic_object_name,
        scene_summary=scene_summary,
        instructions=instructions_str,
        directions_list=directions_list,
    )

    # Combine all frames + masked images
    all_images = all_video_frames + masked_images

    response = call_gemini(prompt, all_images)
    if response is None:
        return None

    answer = response.strip()
    # Try to match with known directions
    for direction in ALL_MOVEMENT_DIRECTIONS:
        if direction.lower() in answer.lower():
            print(f"    Movement direction determined: {direction}")
            return direction

    # If no exact match, return raw (will be used as-is)
    print(f"    Movement direction (raw): {answer}")
    # Try to find closest match
    answer_lower = answer.lower()
    if "left" in answer_lower and "up" in answer_lower:
        return "Moving Left-Up"
    elif "left" in answer_lower and "down" in answer_lower:
        return "Moving Left-Down"
    elif "right" in answer_lower and "up" in answer_lower:
        return "Moving Right-Up"
    elif "right" in answer_lower and "down" in answer_lower:
        return "Moving Right-Down"
    elif "left" in answer_lower:
        return "Moving Left"
    elif "right" in answer_lower:
        return "Moving Right"
    elif "up" in answer_lower:
        return "Moving Up"
    elif "down" in answer_lower:
        return "Moving Down"
    elif "toward" in answer_lower or "closer" in answer_lower:
        return "Moving Toward Camera"
    elif "away" in answer_lower or "farther" in answer_lower:
        return "Moving Away from Camera"
    elif "stationary" in answer_lower or "static" in answer_lower:
        return "Stationary"

    return answer  # return raw if nothing matches


# ============ Gemini: Determine Speed Change ============

def determine_speed_change_gemini(
    all_video_frames: list,
    masked_images: list,
    key_frame_indices: list,
    dynamic_object_name: str,
    scene_summary: str,
) -> str:
    """
    Use Gemini to determine how the speed of the dynamic object changes.
    Returns one of SPEED_OPTIONS_POOL values, or None on failure.
    """
    speed_options_str = "\n".join([f"- {s}" for s in SPEED_OPTIONS_POOL])
    frame_desc = ", ".join([f"{idx}s" for idx in key_frame_indices])

    prompt = '''\
You are given a video as a sequence of {num_video} frames (at 1 fps, so frame N = Ns).
You are also given {num_masked} key frames (at times: {frame_desc}) where a dynamic \
(moving) object '{obj}' is highlighted with a red overlay and contour.

**Scene context:** {scene_summary}

Your task: Analyze how the **speed** of the highlighted dynamic object '{obj}' \
changes across the masked key frames.

IMPORTANT:
- Compare the displacement of the object between consecutive masked frames. \
If the displacement per unit time increases, the object is speeding up; if it \
decreases, the object is slowing down.
- Also consider the apparent size change: an object getting rapidly larger may \
be accelerating toward the camera.
- The camera may be moving too — try to isolate the object's own speed change \
from camera motion.
- Choose the single best description of how the speed varies.

Available speed descriptions:
{speed_options}

Output ONLY one speed description from the list above, nothing else.'''.format(
        num_video=len(all_video_frames),
        num_masked=len(masked_images),
        frame_desc=frame_desc,
        obj=dynamic_object_name,
        scene_summary=scene_summary,
        speed_options=speed_options_str,
    )

    all_images = all_video_frames + masked_images

    response = call_gemini(prompt, all_images)
    if response is None:
        return None

    answer = response.strip()
    # Try exact match first
    for speed_opt in SPEED_OPTIONS_POOL:
        if speed_opt.lower() in answer.lower():
            print(f"    Speed change determined: {speed_opt}")
            return speed_opt

    # Fuzzy matching
    answer_lower = answer.lower()
    if "speed up" in answer_lower or "accelerat" in answer_lower or "faster" in answer_lower:
        return "Speeding Up"
    elif "slow down" in answer_lower or "decelerat" in answer_lower or "slower" in answer_lower:
        return "Slowing Down"
    elif "constant" in answer_lower or "steady" in answer_lower or "uniform" in answer_lower:
        return "Maintaining a Constant Speed"
    elif ("first" in answer_lower and "speed" in answer_lower and "slow" in answer_lower):
        return "First Speeding Up, Then Slowing Down"
    elif ("first" in answer_lower and "slow" in answer_lower and "speed" in answer_lower):
        return "First Slowing Down, Then Speeding Up"
    elif "vary" in answer_lower or "irregular" in answer_lower or "fluctuat" in answer_lower:
        return "Moving at Varying Speeds"

    print(f"    Speed change (raw, no match): {answer}")
    return "Moving at Varying Speeds"  # safe fallback


# ============ Distractor Generation ============

def generate_distractors(correct_direction: str, num_distractors: int = 3) -> list:
    """Generate plausible wrong movement direction options."""
    distractors = []
    used = {correct_direction}

    # Strategy 1: Opposite direction
    if correct_direction in OPPOSITES:
        opp = OPPOSITES[correct_direction]
        if opp not in used:
            distractors.append(opp)
            used.add(opp)

    # Strategy 2: Related directions (e.g., if Moving Left, add Moving Left-Up, etc.)
    related_map = {
        "Moving Left": ["Moving Left-Up", "Moving Left-Down"],
        "Moving Right": ["Moving Right-Up", "Moving Right-Down"],
        "Moving Up": ["Moving Left-Up", "Moving Right-Up"],
        "Moving Down": ["Moving Left-Down", "Moving Right-Down"],
        "Moving Left-Up": ["Moving Left", "Moving Up"],
        "Moving Left-Down": ["Moving Left", "Moving Down"],
        "Moving Right-Up": ["Moving Right", "Moving Up"],
        "Moving Right-Down": ["Moving Right", "Moving Down"],
        "Moving Toward Camera": ["Moving Down", "Moving Up"],
        "Moving Away from Camera": ["Moving Up", "Moving Down"],
        "Stationary": ["Moving Left", "Moving Right"],
    }
    for related in related_map.get(correct_direction, []):
        if len(distractors) >= num_distractors:
            break
        if related not in used:
            distractors.append(related)
            used.add(related)

    # Strategy 3: Random fill
    available = [d for d in ALL_MOVEMENT_DIRECTIONS if d not in used]
    random.shuffle(available)
    while len(distractors) < num_distractors and available:
        distractors.append(available.pop(0))

    return distractors[:num_distractors]


# ============ Question Generation ============

# Diverse question templates grouped by type
QUESTION_STYLE_TEMPLATES = {
    "direct": [
        # Style 1: concise & direct
        "As seen from the camera, in which direction does the {obj} move between {start}s and {end}s?",
        # Style 2: observation-oriented
        "Observing the video from {start}s to {end}s, what is the movement direction of the {obj} as captured by the camera?",
        # Style 3: analytical
        "Analyze the motion of the {obj} from the camera's viewpoint during the interval {start}s to {end}s. Which direction does it move?",
        # Style 4: trajectory-focused
        "From the camera's perspective, describe the trajectory of the {obj} between {start}s and {end}s. In which direction does it primarily travel?",
        # Style 5: scene-setting
        "Between {start}s and {end}s, a {obj} is in motion. From the camera's point of view, which direction does it move?",
        # Style 6: identification
        "Watch the {obj} from {start}s to {end}s. As seen through the camera lens, which direction does it move?",
    ],
    "bbox": [
        "From the perspective of the camera at {start}s, how does the direction of the object with initial bounding box coordinates {bbox} change between {start}s and {end}s?",
        "At {start}s, an object is located within bounding box {bbox}. As seen from the camera, in which direction does this object move by {end}s?",
        "Consider the object at bounding box coordinates {bbox} at time {start}s. From the camera's viewpoint, what is its movement direction through {end}s?",
        "The camera captures an object within bounding box {bbox} at {start}s. By {end}s, in which direction has this object moved as seen in the camera frame?",
    ],
    "distance": [
        "Between {start}s and {end}s, how does the distance between the {obj} and the camera change?",
        "From {start}s to {end}s, does the {obj} move closer to, farther from, or maintain its distance from the camera?",
        "Observing the {obj} from {start}s to {end}s, how does its proximity to the camera change?",
        "As the video progresses from {start}s to {end}s, how does the apparent distance of the {obj} relative to the camera change?",
    ],
    "speed": [
        "How does the speed of the {obj} change between {start}s and {end}s?",
        "Observing the {obj} from {start}s to {end}s, how does its speed vary over this period?",
        "Between {start}s and {end}s, does the {obj} speed up, slow down, or maintain a constant speed?",
        "Assess how the velocity of the {obj} changes during the interval from {start}s to {end}s.",
        "From {start}s to {end}s, what best describes the change in the {obj}'s speed?",
    ],
}

# Distance-specific options
DISTANCE_OPTIONS_MAP = {
    "Moving Toward Camera": "Getting Closer to Camera",
    "Moving Away from Camera": "Getting Farther from Camera",
}

# Speed-specific options pool
SPEED_OPTIONS_POOL = [
    "Speeding Up",
    "Slowing Down",
    "Maintaining a Constant Speed",
    "First Speeding Up, Then Slowing Down",
    "First Slowing Down, Then Speeding Up",
    "Moving at Varying Speeds",
]


def generate_question_with_gemini(
    dynamic_object_name: str,
    start_frame: int,
    end_frame: int,
    options: dict,
    question_type: str,
    bbox_str: str = None,
) -> str:
    """
    Use Gemini to generate a diverse, natural question phrasing for the
    dynamic object movement multiple-choice question.
    question_type: 'direct', 'bbox', 'distance', or 'speed'
    """
    options_str = "\n".join([f"{label}. {text}" for label, text in options.items()])

    if question_type == "bbox":
        prompt = '''\
You are writing a multiple-choice question about a moving object in a video.

Context: A video is given as input (1 fps). We want to ask about the movement direction \
of a dynamic object **as seen from the camera's perspective** during {start}s to {end}s.

The object can be described as: '{obj}'
The object's initial bounding box coordinates (in the model's coordinate space) are: {bbox}

The answer options are:
{options_str}

Please write a single question (in English) that:
- Asks the viewer to identify the movement direction of the object from the camera's viewpoint
- References the bounding box coordinates to specify which object
- Makes clear that the direction is as seen through the camera

Example styles:
- "From the perspective of the camera at {start}s, how does the direction of the object \
with initial bounding box coordinates {bbox} change between {start}s and {end}s?"
- "At {start}s, an object is located at bounding box {bbox}. As seen from the camera, \
in which direction does it move by {end}s?"

Rules:
- Reference the time range and bbox coordinates naturally.
- Keep it to 1-2 sentences.
- Do NOT include the options — just the question text.
- Output ONLY the question text, nothing else.'''.format(
            start=start_frame,
            end=end_frame,
            obj=dynamic_object_name,
            bbox=bbox_str,
            options_str=options_str,
        )

    elif question_type == "distance":
        prompt = '''\
You are writing a multiple-choice question about a moving object's distance change \
relative to the camera in a video.

Context: A video is given as input (1 fps). We want to ask about how the distance \
between '{obj}' and the camera changes during {start}s to {end}s.

The answer options are:
{options_str}

Please write a single question (in English) that asks the viewer to determine how \
the distance/proximity between the '{obj}' and the camera changes during this segment.

Example styles:
- "Between {start}s and {end}s, how does the distance between the {obj} and the camera change?"
- "From {start}s to {end}s, does the {obj} approach or recede from the camera?"
- "Observing the {obj} from {start}s to {end}s, how does its proximity to the camera change?"

Rules:
- Focus on distance/proximity change, not lateral direction.
- Naturally reference the time range and object name.
- Do NOT mention any masks, highlights, or overlays.
- Do NOT include the options — just the question text.
- Keep it to 1-2 sentences.
- Output ONLY the question text, nothing else.'''.format(
            start=start_frame,
            end=end_frame,
            obj=dynamic_object_name,
            options_str=options_str,
        )

    elif question_type == "speed":
        prompt = '''\
You are writing a multiple-choice question about how a moving object's speed varies \
over time in a video.

Context: A video is given as input (1 fps). We want to ask about how the speed of \
'{obj}' changes during {start}s to {end}s.

The answer options are:
{options_str}

Please write a single question (in English) that asks the viewer to assess how the \
speed of the '{obj}' varies over the given time period.

Example styles:
- "How does the speed of the {obj} change between {start}s and {end}s?"
- "Between {start}s and {end}s, does the {obj} speed up, slow down, or move at a constant pace?"
- "Assess how the velocity of the {obj} varies from {start}s to {end}s."
- "Observing the {obj} from {start}s to {end}s, what best describes the change in its speed?"

Rules:
- Focus on speed/velocity change (acceleration, deceleration, constant, varying).
- Naturally reference the time range and object name.
- Do NOT mention any masks, highlights, or overlays.
- Do NOT include the options — just the question text.
- Keep it to 1-2 sentences.
- Output ONLY the question text, nothing else.'''.format(
            start=start_frame,
            end=end_frame,
            obj=dynamic_object_name,
            options_str=options_str,
        )

    else:
        # Direct question type
        style_hints = [
            "a concise and direct style, making clear the direction is from the camera's viewpoint",
            "a descriptive style that sets the scene and asks about movement as seen through the camera",
            "an analytical style asking the viewer to determine the object's trajectory in the camera frame",
            "a conversational tone asking about the direction the object moves from the camera's perspective",
            "a formal cinematography style referencing the camera's field of view",
            "a style focusing on the object's motion path as captured by the camera",
        ]
        chosen_style = random.choice(style_hints)

        prompt = '''\
You are writing a multiple-choice question about a moving object in a video.

Context: A video is given as input (1 fps). We want to ask about the movement direction \
of '{obj}' **as seen from the camera's perspective** during {start}s to {end}s.

The answer options are:
{options_str}

Please write a single question (in English) that asks the viewer to identify \
the movement direction of '{obj}' from the camera's viewpoint. Use {style}.

Rules:
- Make it clear the direction is as observed from the camera / in the camera frame.
- Naturally reference the time range ({start}s to {end}s) and the object name.
- Do NOT mention any masks, highlights, or overlays.
- Do NOT include the options — just the question text.
- Keep it to 1-2 sentences.
- Output ONLY the question text, nothing else.'''.format(
            start=start_frame,
            end=end_frame,
            obj=dynamic_object_name,
            options_str=options_str,
            style=chosen_style,
        )

    response = call_gemini(prompt)
    if response and len(response.strip()) > 10:
        question_body = response.strip().strip('"').strip("'")
        return f"{question_body}\n\n{options_str}"

    # Fallback: use a random template
    templates = QUESTION_STYLE_TEMPLATES.get(question_type, QUESTION_STYLE_TEMPLATES["direct"])
    tmpl = random.choice(templates)
    fallback_q = tmpl.format(
        obj=dynamic_object_name, start=start_frame, end=end_frame,
        bbox=bbox_str or "",
    )
    return f"{fallback_q}\n\n{options_str}"


# ============ CoT Format Validation ============

def validate_cot_format(response: str, expected_image_count: int) -> bool:
    """
    Validate that the CoT response follows the required format:
    - <thinking> ... </thinking> with substantial reasoning and <output_image> placeholders
    - <answer> ... </answer> after </thinking>
    - <thinking> must NOT contain <answer> tags
    - <output_image> tags must appear exactly expected_image_count times
    - <output_image> tags should be grouped together
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

    # Check that <output_image> tags are grouped
    first_tag_pos = think_content.index("<output_image>")
    last_tag_pos = think_content.rindex("<output_image>")
    tag_span = last_tag_pos - first_tag_pos
    max_span = expected_image_count * 150
    if tag_span > max_span:
        return False

    return True


# ============ CoT Generation ============

def generate_cot_with_gemini(
    all_video_frames: list,
    masked_images: list,
    key_frame_indices: list,
    dynamic_object_name: str,
    correct_direction: str,
    correct_label: str,
    options: dict,
    start_frame: int,
    end_frame: int,
    total_frames: int,
    question_text: str,
) -> str:
    """
    Call Gemini to generate a detailed CoT reasoning for dynamic object movement.

    Input images to Gemini (in order):
      1. Video Frames: ALL frames of the entire video
      2. Masked Frames: key frames with dynamic object highlighted in red
         -- "mental imagery", referenced via <output_image>

    Returns the CoT string, or None if all retries fail.
    """
    max_attempts = 10
    num_video = len(all_video_frames)
    num_masked = len(masked_images)

    masked_frame_desc = ", ".join([f"{idx}s" for idx in key_frame_indices])

    instruction_text = '''\
You are an advanced multimodal reasoning assistant. Your task is to generate \
a comprehensive, step-by-step Chain of Thought (CoT) reasoning process that \
leads to the correct option for a given multiple-choice question about a \
dynamic object's movement (as observed from the camera's viewpoint).

**Input Data Provided to You:**
1.  **Video Frames:** The complete video as a sequence of {num_video} frames \
(from 0s to {total_frames}s, at 1 fps).
2.  **Masked Frames:** {num_masked} key frames (at {masked_frame_desc}) where \
the dynamic object '{obj}' is highlighted in red overlay and red contour. These \
show the object's position at specific moments.
3.  **Multiple Choice Question:**
{question_text}
4.  **Correct Answer:** {correct_label}. {correct_direction}

**Your Goal:**
You must engineer the reasoning process. You need to output a thought \
process that logically uses the visual inputs to derive the provided Correct Answer.

**Critical Rules for `<thinking>` and `<output_image>`:**
1.  **Mental Imagery as Output (NOT Input):** The Masked Frames are NOT part of \
your input -- they represent the **output of your own mental imagination**. During \
reasoning, you should describe a process where you *mentally visualize* highlighting \
the dynamic object '{obj}' in your mind, imagining where it appears across frames. \
Each `<output_image>` tag represents one such mental image you produce in your mind's \
eye, where you picture the object highlighted (as shown in the Masked Frames). \
Think of it as: "In my mind, I picture the '{obj}' highlighted across the key moments, \
forming a progression from <output_image> to <output_image> to <output_image>."
2.  **Strict Usage of `<output_image>`:**
    *   Each `<output_image>` is a **mental image you generate** -- your imagination \
of the dynamic object '{obj}' highlighted in red at a specific moment in time. \
You **must** use the placeholder `<output_image>` to represent these imagined frames.
    *   **Quantity & Order:** You must use exactly {num_masked} `<output_image>` tags \
(one per imagined frame), strictly in chronological order.
    *   **Grouping:** You must present these tags **all together in one sentence** \
to show the mental progression.
        *   *Correct Format:* "...I mentally visualize tracking the object from \
<output_image> to <output_image> to <output_image>, where the '{obj}' is highlighted \
in each imagined frame to track its position change."
3.  **Negative Constraint (Important):**
    *   **Do NOT** use the `<output_image>` tag again in your subsequent analysis text.
    *   When analyzing the shift, refer to them using natural language like \
"In the first frame," "In the final view," etc.
4.  **Reasoning Flow:**
    *   **Scene Overview:** Briefly describe the overall video scene and identify \
the dynamic object '{obj}'.
    *   **Temporal Localization:** Identify the time segment ({start_frame}s-{end_frame}s) \
the question focuses on.
    *   **Strategy:** State the need to track the dynamic object's position across \
multiple frames to determine its movement as seen from the camera.
    *   **Visualization:** Describe your mental process of imagining the object \
highlighted, and present the `<output_image>` sequence as your own mental imagery \
output (strictly matching count of {num_masked}).
    *   **Position Analysis:** Describe the object's position change observed in \
that sequence (using text only, no tags). Note the shift in screen coordinates \
and/or apparent size change (for depth).
    *   **Camera Compensation (if needed):** If the camera is also moving, account \
for how camera motion affects the apparent object position on screen. Separate \
camera motion from the object's own movement.
    *   **Conclusion:** Determine the object's movement direction / distance change \
as observed from the camera, and match it to the correct option.

**Output Format:**
<thinking>
[Detailed reasoning ...]
</thinking>
<answer>
[The Correct Option, e.g., "A. Moving Left"]
</answer>

**Example:**
(Suppose the video has 30 frames covering 0s-30s, and we track a person walking.)
<thinking>
The question asks about the movement direction of the person between 5s and 25s \
as seen from the camera. Looking through the full video, I can see a street scene \
with a person walking along the sidewalk.

Focusing on the segment from 5s to 25s, the person appears to be in motion while \
the camera remains relatively stable. To determine the exact movement direction \
from the camera's viewpoint, I need to track the person's position across frames.

In my mind, I imagine highlighting the person with a red overlay across the key \
moments to track their trajectory. I mentally visualize the progression: from \
<output_image> to <output_image> to <output_image> to <output_image>, where the \
person is highlighted in each imagined frame.

Analyzing this mental sequence, I observe that in the first imagined frame, the \
person is positioned near the right side of the camera frame. In subsequent frames, \
the person gradually shifts toward the left side. By the final frame, the person has \
moved significantly to the left of the frame.

Since the camera appears stable during this segment, the leftward shift on screen \
directly indicates the person is moving to the left from the camera's perspective.

This matches Option C.
</thinking>
<answer>C. Moving Left</answer>

The images follow below (first {num_video} are Video Frames of the complete video, \
next {num_masked} are Masked Frames for the target segment).'''.format(
        num_video=num_video,
        num_masked=num_masked,
        total_frames=total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        masked_frame_desc=masked_frame_desc,
        obj=dynamic_object_name,
        question_text=question_text,
        correct_label=correct_label,
        correct_direction=correct_direction,
    )

    for attempt in range(max_attempts):
        model = AVAILABLE_MODELS[attempt % len(AVAILABLE_MODELS)]
        print(f"    CoT attempt {attempt + 1}/{max_attempts} (model={model})...")

        try:
            user_content = [{"type": "text", "text": instruction_text}]

            for img_path in all_video_frames:
                image_data, mime_type = encode_image(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                })

            user_content.append({
                "type": "text",
                "text": "--- The following {n} images are Masked Frames (your mental imagery of tracking the dynamic object highlighted in red, use <output_image> for each): ---".format(n=num_masked)
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
                    "M-TraceId": f"4dthinker_dyncot_{int(time.time())}_{random.randint(0, 9999)}"
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

    print(f"    FAILED: All {max_attempts} CoT attempts exhausted, skipping")
    return None


# ============ Utility Functions ============

def get_all_frame_paths(frames_dir: str) -> list:
    """Get sorted list of ALL frame image paths in the frames directory."""
    if not os.path.isdir(frames_dir):
        return []
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    return [os.path.join(frames_dir, f) for f in frame_files]


def get_dynamic_mask_paths(masks_dir: str) -> list:
    """Get sorted list of dynamic mask .npy file paths."""
    if not os.path.isdir(masks_dir):
        return []
    mask_files = sorted([
        f for f in os.listdir(masks_dir)
        if f.endswith('.npy')
    ])
    return [os.path.join(masks_dir, f) for f in mask_files]


def extract_frame_index(filename: str) -> int:
    """Extract frame index from filename like 'frame_0009.npy' or 'frame_0009.jpg'."""
    name = os.path.splitext(os.path.basename(filename))[0]
    # Handle patterns like frame_0009 or frame_0009_mask
    parts = name.split('_')
    for part in parts[1:]:
        if part.isdigit():
            return int(part)
    return -1


def select_mask_frames(mask_paths: list, min_count: int = 4, max_count: int = 8) -> list:
    """
    Select 4-8 mask paths, preferring first and last frames, then evenly
    sampling intermediate ones.
    Returns selected mask paths sorted by frame index.
    """
    if len(mask_paths) <= min_count:
        return mask_paths  # Use all if we have fewer than min

    # Always include first and last
    selected = [mask_paths[0], mask_paths[-1]]
    remaining = mask_paths[1:-1]

    # Determine how many more to sample
    target_count = min(max_count, len(mask_paths))
    num_additional = target_count - 2  # already have first and last

    if num_additional > 0 and remaining:
        if len(remaining) <= num_additional:
            selected.extend(remaining)
        else:
            # Evenly sample from remaining
            step = len(remaining) / (num_additional + 1)
            indices = [int(round(step * (i + 1))) for i in range(num_additional)]
            indices = [min(i, len(remaining) - 1) for i in indices]
            indices = sorted(set(indices))
            for idx in indices:
                selected.append(remaining[idx])

    # Sort by frame index
    selected.sort(key=lambda p: extract_frame_index(p))
    return selected


# ============ Main Processing for Single Video ============

def process_single_data(video_id: str):
    """
    Process a single video: generate dynamic object movement QA.
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

    dynamic_object = data.get("dynamic_object", "")
    instructions = data.get("instructions", {})
    caption = data.get("caption", {})
    scene_summary = caption.get("SceneSummary", "")

    frames_dir = os.path.join(data_dir, "frames")
    masks_dynamic_dir = os.path.join(data_dir, "masks_dynamic")

    # ---- Step 1: Check masks_dynamic has content ----
    if not os.path.isdir(masks_dynamic_dir):
        print(f"SKIP: No masks_dynamic directory for {video_id}")
        return []

    all_mask_paths = get_dynamic_mask_paths(masks_dynamic_dir)
    if len(all_mask_paths) == 0:
        print(f"SKIP: masks_dynamic is empty for {video_id}")
        return []

    print(f"Found {len(all_mask_paths)} dynamic masks for {video_id}")

    if not dynamic_object or dynamic_object == "null":
        print(f"WARNING: No dynamic_object name, will rely on Gemini to identify it")
        dynamic_object = "moving object"

    # ---- Step 2: Select 4-8 mask frames ----
    selected_masks = select_mask_frames(all_mask_paths, min_count=4, max_count=8)
    print(f"Selected {len(selected_masks)} mask frames for verification")

    # Get corresponding frame paths and frame indices
    selected_frame_indices = [extract_frame_index(p) for p in selected_masks]
    selected_frame_paths = [
        os.path.join(frames_dir, f"frame_{idx:04d}.jpg") for idx in selected_frame_indices
    ]

    # Verify frame images exist
    valid_pairs = []
    for mask_p, frame_p, idx in zip(selected_masks, selected_frame_paths, selected_frame_indices):
        if os.path.exists(frame_p):
            valid_pairs.append((mask_p, frame_p, idx))
        else:
            print(f"  WARNING: Frame image missing for mask at index {idx}")

    if len(valid_pairs) < 2:
        print(f"SKIP: Too few valid frame-mask pairs ({len(valid_pairs)}) for {video_id}")
        return []

    # Create overlay images
    overlay_dir = os.path.join(data_dir, "masked_dynamic_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    overlay_info = []  # (frame_idx, overlay_path, original_frame_path, mask_path)
    for mask_p, frame_p, idx in valid_pairs:
        overlay_path = os.path.join(overlay_dir, f"dyn_f{idx:04d}.jpg")
        if create_masked_overlay(frame_p, mask_p, overlay_path):
            overlay_info.append((idx, overlay_path, frame_p, mask_p))
        else:
            print(f"  WARNING: Failed to create overlay for frame {idx}")

    if len(overlay_info) < 2:
        print(f"SKIP: Too few valid overlays ({len(overlay_info)}) for {video_id}")
        return []

    print(f"Created {len(overlay_info)} overlay images")

    # ---- Step 3: Gemini verify same dynamic object ----
    overlay_paths = [info[1] for info in overlay_info]
    print(f"Verifying dynamic object with Gemini ({len(overlay_paths)} frames)...")

    gemini_result = check_dynamic_object_gemini(overlay_paths, dynamic_object)
    valid_indices = gemini_result["valid_indices"]
    identified_object = gemini_result["object_name"]

    if len(valid_indices) == 0:
        print(f"SKIP: No valid dynamic object frames identified by Gemini for {video_id}")
        return []

    # Filter to keep only valid frames
    filtered_overlay_info = [overlay_info[i] for i in valid_indices if i < len(overlay_info)]
    if len(filtered_overlay_info) < 2:
        print(f"SKIP: After filtering, only {len(filtered_overlay_info)} frames remain for {video_id}")
        return []

    print(f"After Gemini filtering: {len(filtered_overlay_info)} valid frames, "
          f"object='{identified_object}'")

    # Use the identified object name if available
    if identified_object and identified_object != dynamic_object:
        print(f"  Using Gemini-identified object name: '{identified_object}'")
        dynamic_object = identified_object

    # ---- Step 4: Get all video frames ----
    all_original_frame_paths = get_all_frame_paths(frames_dir)
    total_frames = len(all_original_frame_paths)
    print(f"Total frames in video: {total_frames}")

    # Determine time range from filtered frames
    filtered_frame_indices = [info[0] for info in filtered_overlay_info]
    start_frame = min(filtered_frame_indices)
    end_frame = max(filtered_frame_indices)
    filtered_overlay_paths = [info[1] for info in filtered_overlay_info]

    # ---- Step 5: Determine movement direction with Gemini ----
    print(f"Determining movement direction with Gemini...")
    movement_direction = determine_movement_direction_gemini(
        all_video_frames=all_original_frame_paths,
        masked_images=filtered_overlay_paths,
        key_frame_indices=filtered_frame_indices,
        dynamic_object_name=dynamic_object,
        scene_summary=scene_summary,
        instructions=instructions,
    )

    if movement_direction is None:
        print(f"SKIP: Failed to determine movement direction for {video_id}")
        return []

    print(f"Movement direction GT: {movement_direction}")

    # ---- Step 6: Compute bbox from first mask (for bbox-style questions) ----
    first_mask_path = filtered_overlay_info[0][3]  # mask_path of first frame
    first_frame_path = filtered_overlay_info[0][2]  # original frame path
    first_frame_idx = filtered_overlay_info[0][0]

    first_mask = np.load(first_mask_path)
    if first_mask.max() > 1:
        first_mask_bool = first_mask > 127
    else:
        first_mask_bool = first_mask > 0

    bbox_raw = bbox_from_mask(first_mask_bool)

    frame_img = cv2.imread(first_frame_path)
    bbox_converted = None
    bbox_str = None
    if bbox_raw is not None and frame_img is not None:
        orig_h, orig_w = frame_img.shape[:2]
        try:
            bbox_converted = convert_to_qwen25vl_format(bbox_raw, orig_h, orig_w)
            bbox_str = f"[{bbox_converted[0]},{bbox_converted[1]},{bbox_converted[2]},{bbox_converted[3]}]"
            print(f"  Bbox raw: {bbox_raw} -> converted: {bbox_converted} "
                  f"(frame size: {orig_w}x{orig_h})")
        except ValueError as e:
            print(f"  WARNING: Bbox conversion failed: {e}")

    # ---- Step 7: Randomly select 2 question types ----
    # Available types and their base weights: direct=0.3, bbox=0.4, distance=0.2, speed=0.1
    # Some types require prerequisites; if not met, their weight is redistributed.
    candidate_types = []
    candidate_weights = []

    # direct: always available
    candidate_types.append("direct")
    candidate_weights.append(0.32)

    # bbox: requires valid bbox
    if bbox_converted is not None:
        candidate_types.append("bbox")
        candidate_weights.append(0.45)

    # distance: always available (direction GT maps to a distance answer)
    candidate_types.append("distance")
    candidate_weights.append(0.2)

    # speed: requires non-stationary object
    if movement_direction != "Stationary":
        candidate_types.append("speed")
        candidate_weights.append(0.03)

    # Select 2 unique types via weighted sampling without replacement
    num_to_select = min(2, len(candidate_types))
    question_types_to_generate = []
    remaining_types = list(candidate_types)
    remaining_weights = list(candidate_weights)

    for _ in range(num_to_select):
        # Normalize weights
        total_w = sum(remaining_weights)
        probs = [w / total_w for w in remaining_weights]
        chosen_idx = random.choices(range(len(remaining_types)), weights=probs, k=1)[0]
        question_types_to_generate.append(remaining_types[chosen_idx])
        remaining_types.pop(chosen_idx)
        remaining_weights.pop(chosen_idx)

    print(f"Question types to generate: {question_types_to_generate}")

    # Determine speed change with Gemini only if "speed" was selected
    speed_change = None
    if "speed" in question_types_to_generate:
        print(f"Determining speed change with Gemini...")
        speed_change = determine_speed_change_gemini(
            all_video_frames=all_original_frame_paths,
            masked_images=filtered_overlay_paths,
            key_frame_indices=filtered_frame_indices,
            dynamic_object_name=dynamic_object,
            scene_summary=scene_summary,
        )
        if speed_change is None:
            # Failed to determine speed, replace "speed" with another type
            print(f"  WARNING: Speed determination failed, replacing with fallback")
            question_types_to_generate.remove("speed")
            fallback_candidates = [t for t in candidate_types
                                   if t not in question_types_to_generate and t != "speed"]
            if fallback_candidates:
                question_types_to_generate.append(random.choice(fallback_candidates))
        else:
            print(f"Speed change GT: {speed_change}")

    # ---- Step 8: Generate questions for each selected type ----
    results = []

    for q_type in question_types_to_generate:
        print(f"\n  --- Generating '{q_type}' question ---")

        # Build options specific to this question type
        if q_type == "distance":
            # Distance questions use proximity-based options
            distance_options_pool = [
                "Getting Closer to Camera",
                "Getting Farther from Camera",
                "Maintaining Roughly the Same Distance",
            ]
            # Map the movement direction to a distance answer
            if movement_direction == "Moving Toward Camera":
                correct_dist = "Getting Closer to Camera"
            elif movement_direction == "Moving Away from Camera":
                correct_dist = "Getting Farther from Camera"
            else:
                correct_dist = "Maintaining Roughly the Same Distance"

            other_dist = [d for d in distance_options_pool if d != correct_dist]
            extra_distractors = [
                "First Getting Closer, Then Farther",
                "First Getting Farther, Then Closer",
            ]
            other_dist.append(random.choice(extra_distractors))

            all_dist_options = [correct_dist] + other_dist[:3]
            random.shuffle(all_dist_options)
            correct_idx = all_dist_options.index(correct_dist)

            option_labels = ["A", "B", "C", "D"]
            options = {option_labels[i]: opt for i, opt in enumerate(all_dist_options)}
            correct_label = option_labels[correct_idx]
            correct_answer_text = correct_dist

        elif q_type == "speed":
            # Speed questions use speed-variation options
            correct_speed = speed_change  # already determined by Gemini
            other_speeds = [s for s in SPEED_OPTIONS_POOL if s != correct_speed]
            random.shuffle(other_speeds)
            distractors_speed = other_speeds[:3]

            all_speed_options = [correct_speed] + distractors_speed
            random.shuffle(all_speed_options)
            correct_idx = all_speed_options.index(correct_speed)

            option_labels = ["A", "B", "C", "D"]
            options = {option_labels[i]: opt for i, opt in enumerate(all_speed_options)}
            correct_label = option_labels[correct_idx]
            correct_answer_text = correct_speed

        else:
            # Direction-based questions (direct / bbox)
            distractors = generate_distractors(movement_direction, num_distractors=3)

            all_options_list = [movement_direction] + distractors
            random.shuffle(all_options_list)
            correct_idx = all_options_list.index(movement_direction)

            option_labels = ["A", "B", "C", "D"]
            options = {option_labels[i]: opt for i, opt in enumerate(all_options_list)}
            correct_label = option_labels[correct_idx]
            correct_answer_text = movement_direction

        # Generate question text
        print(f"  Generating {q_type} question with Gemini...")
        question_text = generate_question_with_gemini(
            dynamic_object_name=dynamic_object,
            start_frame=start_frame,
            end_frame=end_frame,
            options=options,
            question_type=q_type,
            bbox_str=bbox_str,
        )

        # Generate CoT
        print(f"  Generating CoT for {q_type} question...")
        cot = generate_cot_with_gemini(
            all_video_frames=all_original_frame_paths,
            masked_images=filtered_overlay_paths,
            key_frame_indices=filtered_frame_indices,
            dynamic_object_name=dynamic_object,
            correct_direction=correct_answer_text,
            correct_label=correct_label,
            options=options,
            start_frame=start_frame,
            end_frame=end_frame,
            total_frames=total_frames,
            question_text=question_text,
        )

        if cot is not None:
            result = {
                "id": f"{video_id}_dynamic_{q_type}",
                "question": question_text.strip(),
                "options": options,
                "correct_answer": f"{correct_label}. {correct_answer_text}",
                "cot": cot,
                "mask_image_paths": filtered_overlay_paths,
                "original_image_paths": all_original_frame_paths,
            }
            results.append(result)
            print(f"  SUCCESS: {q_type} question generated")
        else:
            print(f"  SKIP: CoT generation failed for {q_type} question")

    return results


# ============ Batch Processing ============

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
    """Load already-processed video_id prefixes from existing JSONL for resume.
    Extracts video_id from the 'id' field (format: '{video_id}_dynamic_{type}').
    """
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
                    item_id = obj.get("id", "")
                    # id format: "{video_id}_dynamic_{type}"
                    # video_id is UUID, so split on "_dynamic_"
                    if "_dynamic_" in item_id:
                        vid = item_id.rsplit("_dynamic_", 1)[0]
                    else:
                        vid = item_id
                    if vid:
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

        return video_id, num_results

    except Exception as e:
        with progress_info["lock"]:
            progress_info["done"] += 1
            done = progress_info["done"]
            total = progress_info["total"]
        print(f"\n[Progress {done}/{total}] video={video_id} -> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return video_id, 0


def main():
    # ---- Configuration ----
    MAX_WORKERS = 16
    output_file = os.path.join(BASE_DIR, "dynamic_object_qa_all.jsonl")

    # Support optional single video_id mode via command line
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        # Single video mode
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
        for r in results:
            print(f"  - {r['id']} -> {r['correct_answer']}")
        return

    # ---- Batch mode: process ALL video directories ----
    print(f"{'#'*60}")
    print(f"Scanning all video directories in {DATA_DIR}...")
    print(f"{'#'*60}")

    all_video_ids = get_all_video_ids()
    print(f"Found {len(all_video_ids)} video directories with data.jsonl")

    # Filter: only keep videos that have masks_dynamic with content
    eligible_ids = []
    for vid in all_video_ids:
        masks_dir = os.path.join(DATA_DIR, vid, "masks_dynamic")
        if os.path.isdir(masks_dir) and any(f.endswith('.npy') for f in os.listdir(masks_dir)):
            eligible_ids.append(vid)

    print(f"Found {len(eligible_ids)} videos with non-empty masks_dynamic")

    # Load already-completed video IDs for resume
    completed_ids = load_completed_ids(output_file)
    if completed_ids:
        print(f"Found {len(completed_ids)} already-processed videos, skipping them.")

    pending_ids = [vid for vid in eligible_ids if vid not in completed_ids]
    print(f"Pending: {len(pending_ids)} videos to process")

    if not pending_ids:
        print("All eligible videos already processed. Nothing to do.")
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
