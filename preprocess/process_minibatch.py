#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# Set environment variables before importing torch/triton
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache_' + str(os.getpid())

# Create gcc wrapper script to add -std=gnu99 flag
_gcc_wrapper_path = '/tmp/gcc_c99_wrapper.sh'
with open(_gcc_wrapper_path, 'w') as _f:
    _f.write('#!/bin/bash\nexec gcc -std=gnu99 "$@"\n')
os.chmod(_gcc_wrapper_path, 0o755)
os.environ['CC'] = _gcc_wrapper_path

import json
import cv2
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import traceback
from typing import List, Dict, Optional
import torch

# ---------------- Configuration ----------------
BATCH_SIZE = 100  # Max number of videos to process per run

# Path configuration
SAM3_PATH = "./sam3/sam3-main"
BASE_VIDEO_DIR = "./data/SpatialVID/videos"
BASE_ANNO_DIR = "./data/SpatialVID/annotations"
METADATA_CSV = "./data/SpatialVID/SpatialVID_metadata.csv"
OUTPUT_BASE_DIR = "./output"
SAM3_CHECKPOINT = "./sam3/models/sam3.pt"
MAX_FRAMES = 50

COMPLETED_FILE = os.path.join(OUTPUT_BASE_DIR, "completed_videos.txt")
RESUME_FILE = os.path.join(OUTPUT_BASE_DIR, "resume_position.json")

# OpenAI
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# ---------------- SAM3 Setup ----------------
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

try:
    from sam3.model_builder import build_sam3_video_predictor
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM3 not available: {e}")
    SAM3_AVAILABLE = False

# ---------------- Global Variables ----------------
completed_videos = set()
predictor = None  # Global SAM3 predictor (singleton)

# Hardware detection
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Initialization: detected {NUM_GPUS} GPUs, using single-thread mode")
print(f"Strategy: process {BATCH_SIZE} videos per run then stop")

# ---------------- Helper Functions ----------------

def load_progress():
    global completed_videos
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, 'r') as f:
            completed_videos = set(line.strip() for line in f if line.strip())
        print(f"Loaded progress: {len(completed_videos)} completed videos")
    else:
        completed_videos = set()

def save_completed_video(video_id: str):
    completed_videos.add(video_id)
    with open(COMPLETED_FILE, 'a') as f:
        f.write(f"{video_id}\n")

def is_video_completed(video_id: str) -> bool:
    return video_id in completed_videos

def get_all_videos():
    video_list = []
    if not os.path.exists(BASE_VIDEO_DIR):
        print(f"Error: Video dir not found: {BASE_VIDEO_DIR}")
        return []

    for group_dir in sorted(Path(BASE_VIDEO_DIR).iterdir()):
        if not group_dir.is_dir(): continue
        for video_file in sorted(group_dir.glob("*.mp4")):
            video_list.append({
                'group': group_dir.name,
                'video_id': video_file.stem,
                'video_path': str(video_file)
            })
    return video_list

def load_resume_position():
    """Load last processing position."""
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, 'r') as f:
                data = json.load(f)
                return data.get('group'), data.get('video_id')
        except Exception as e:
            print(f"Warning: Failed to load resume position: {e}")
    return None, None

def save_resume_position(group: str, video_id: str):
    """Save current processing position."""
    try:
        with open(RESUME_FILE, 'w') as f:
            json.dump({'group': group, 'video_id': video_id}, f)
    except Exception as e:
        print(f"Warning: Failed to save resume position: {e}")

def get_videos_from_position(all_videos, resume_group=None, resume_video_id=None):
    """Get video list starting from specified position."""
    if resume_group is None or resume_video_id is None:
        return all_videos

    start_idx = 0
    for idx, video in enumerate(all_videos):
        if video['group'] == resume_group and video['video_id'] == resume_video_id:
            start_idx = idx + 1
            break
        elif video['group'] > resume_group:
            start_idx = idx
            break

    return all_videos[start_idx:]

def load_metadata(metadata_csv):
    try:
        df = pd.read_csv(metadata_csv)
        df['video_id'] = df['video path'].apply(lambda x: Path(x).stem)
        return df.set_index('video_id')
    except Exception as e:
        print(f"Warning: Failed to load metadata CSV: {e}")
        return None

def extract_frames_from_video(video_path, indexes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_paths = []
    for frame_seq, frame_idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"frame_{frame_seq:04d}.jpg"
            p = os.path.join(output_dir, frame_filename)
            cv2.imwrite(p, frame)
            frame_paths.append(p)
    cap.release()
    return frame_paths

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def identify_objects_with_gpt(frame_paths, max_retry=3):
    # Sample frames to reduce payload
    sample_indices = np.linspace(0, len(frame_paths)-1, min(10, len(frame_paths)), dtype=int)
    sampled_frames = [frame_paths[i] for i in sample_indices]

    content = [{
        "type": "text",
        "text": "Analyze this video sequence and identify: 1) One object that is clearly moving/changing (dynamic object). 2) One object that remains static (static object). Both must be prominent. If none, return 'null'. Format JSON: {\"dynamic_object\": \"label\", \"static_object\": \"label\"}"
    }]

    for fp in sampled_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(fp)}"}
        })

    for attempt in range(max_retry):
        try:
            result = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": content}],
                stream=False
            )
            txt = result.choices[0].message.content
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0].strip()
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0].strip()

            return json.loads(txt)
        except Exception as e:
            if attempt == max_retry - 1:
                print(f"GPT Error: {e}")
                return {'dynamic_object': None, 'static_object': None}
            time.sleep(1)

def is_oom_error(e):
    msg = str(e).lower()
    return 'out of memory' in msg or 'oom' in msg or 'cudnn error' in msg

def init_sam3_predictor():
    """Initialize SAM3 model (singleton pattern)."""
    global predictor
    if predictor is not None:
        return predictor

    if not SAM3_AVAILABLE or NUM_GPUS == 0:
        return None

    print("Loading SAM3 model...")
    torch.cuda.set_device(0)
    predictor = build_sam3_video_predictor(checkpoint_path=SAM3_CHECKPOINT)
    torch.cuda.synchronize()
    print("SAM3 model loaded")
    return predictor

# ---------------- Core Processing Logic ----------------

def extract_masks_with_sam3(frame_dir, dynamic_object, static_object, output_dir):
    """Extract masks using SAM3."""
    os.makedirs(os.path.join(output_dir, "masked_frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_dynamic"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_static"), exist_ok=True)

    sam3_predictor = init_sam3_predictor()
    if sam3_predictor is None:
        return [], [], []

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    num_frames = len(frame_files)

    masked_paths, dyn_paths, sta_paths = [], [], []
    dyn_masks_list, sta_masks_list = [], []

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dynamic_session, static_session = None, None

        try:
            # 1. Start Sessions
            if dynamic_object and dynamic_object != 'null':
                res = sam3_predictor.handle_request(dict(type="start_session", resource_path=frame_dir))
                dynamic_session = res.get("session_id")

            if static_object and static_object != 'null':
                res = sam3_predictor.handle_request(dict(type="start_session", resource_path=frame_dir))
                static_session = res.get("session_id")

            # 2. Process Frames
            for i in range(num_frames):
                # Dynamic
                dm = None
                if dynamic_session:
                    res = sam3_predictor.handle_request(dict(
                        type="add_prompt", session_id=dynamic_session, frame_index=i, text=dynamic_object
                    ))
                    if "outputs" in res:
                        mask = res["outputs"].get("out_binary_masks")
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        if mask is not None and mask.size > 0:
                            dm = mask[0] if len(mask.shape) == 3 else mask
                    dyn_masks_list.append(dm)

                # Static
                sm = None
                if static_session:
                    res = sam3_predictor.handle_request(dict(
                        type="add_prompt", session_id=static_session, frame_index=i, text=static_object
                    ))
                    if "outputs" in res:
                        mask = res["outputs"].get("out_binary_masks")
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        if mask is not None and mask.size > 0:
                            sm = mask[0] if len(mask.shape) == 3 else mask
                    sta_masks_list.append(sm)

        finally:
            # Cleanup sessions
            for sid in [dynamic_session, static_session]:
                if sid:
                    try:
                        sam3_predictor.handle_request(dict(type="stop_session", session_id=sid))
                    except:
                        pass

    # 3. Post-process
    while len(dyn_masks_list) < len(frame_files):
        dyn_masks_list.append(None)
    while len(sta_masks_list) < len(frame_files):
        sta_masks_list.append(None)

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Save Dynamic Mask
        mask = dyn_masks_list[i]
        d_path = ""
        if mask is not None:
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
            mask_u8 = (mask * 255).astype(np.uint8)
            save_p = os.path.join(output_dir, "masks_dynamic", frame_file.replace('.jpg', '.npy'))
            np.save(save_p, mask_u8)
            d_path = save_p
            mask_3ch = np.stack([np.zeros_like(mask_u8), np.zeros_like(mask_u8), mask_u8], axis=2)
            frame = cv2.addWeighted(frame, 1.0, mask_3ch, 0.5, 0)
        dyn_paths.append(d_path)

        # Save Static Mask
        mask = sta_masks_list[i]
        s_path = ""
        if mask is not None:
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
            mask_u8 = (mask * 255).astype(np.uint8)
            save_p = os.path.join(output_dir, "masks_static", frame_file.replace('.jpg', '.npy'))
            np.save(save_p, mask_u8)
            s_path = save_p
            mask_3ch = np.stack([mask_u8, np.zeros_like(mask_u8), np.zeros_like(mask_u8)], axis=2)
            frame = cv2.addWeighted(frame, 1.0, mask_3ch, 0.5, 0)
        sta_paths.append(s_path)

        # Save Overlay
        ov_path = os.path.join(output_dir, "masked_frames", frame_file)
        cv2.imwrite(ov_path, frame)
        masked_paths.append(ov_path)

    return masked_paths, dyn_paths, sta_paths

def process_single_video(video_info, metadata_df):
    """Process a single video end-to-end."""
    group = video_info['group']
    video_id = video_info['video_id']
    video_path = video_info['video_path']
    output_dir = os.path.join(OUTPUT_BASE_DIR, "processed_all", group, video_id)

    try:
        # Check files
        anno_dir = os.path.join(BASE_ANNO_DIR, group, video_id)
        if not os.path.exists(anno_dir):
            return None

        # Load meta
        with open(os.path.join(anno_dir, "indexes.txt"), 'r') as f:
            indexes = [list(map(int, line.strip().split())) for line in f if not line.startswith('#') and len(line.split())==2]

        # Extract frames
        frames_dir = os.path.join(output_dir, "frames")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        frame_paths = extract_frames_from_video(video_path, indexes, frames_dir)
        if not frame_paths or len(frame_paths) > MAX_FRAMES:
            return None

        # GPT object identification
        objs = identify_objects_with_gpt(frame_paths)

        # SAM3 mask extraction
        masked_paths, d_masks, s_masks = extract_masks_with_sam3(
            frames_dir, objs.get('dynamic_object'), objs.get('static_object'), output_dir
        )

        # Save Metadata
        final_data = {
            'video_id': video_id,
            'dynamic_object': objs.get('dynamic_object'),
            'static_object': objs.get('static_object'),
            'masked_frame_paths': masked_paths,
        }

        # Load extra jsons
        for fname in ["instructions.json", "caption.json"]:
            p = os.path.join(anno_dir, fname)
            if os.path.exists(p):
                with open(p) as f:
                    final_data[fname.split('.')[0]] = json.load(f)

        # Write Result
        with open(os.path.join(output_dir, "data.jsonl"), 'w') as f:
            json.dump(final_data, f)
            f.write('\n')

        save_completed_video(video_id)
        return video_id

    except Exception as e:
        if is_oom_error(e):
            print(f"OOM Error processing {video_id}. Will be retried in next batch.")
            torch.cuda.empty_cache()
        else:
            print(f"Error processing {video_id}: {e}")
            traceback.print_exc()
        return None

def main():
    load_progress()
    metadata_df = load_metadata(METADATA_CSV)
    all_videos = get_all_videos()

    # Load resume position
    resume_group, resume_video_id = load_resume_position()
    if resume_group and resume_video_id:
        print(f"Resuming from: group={resume_group}, video_id={resume_video_id}")
        all_videos = get_videos_from_position(all_videos, resume_group, resume_video_id)

    # Filter out completed videos
    remaining_videos = [v for v in all_videos if not is_video_completed(v['video_id'])]

    if not remaining_videos:
        print("All videos processed!")
        if os.path.exists(RESUME_FILE):
            os.remove(RESUME_FILE)
        sys.exit(0)

    # Process BATCH_SIZE videos per run
    batch_size = min(BATCH_SIZE, len(remaining_videos))
    current_batch = remaining_videos[:batch_size]

    already_processed = len(completed_videos)
    print(f"\nProgress: {already_processed} videos completed so far")
    print(f"Current batch: {len(current_batch)} videos (remaining: {len(remaining_videos)})")

    # Single-threaded processing loop
    success_count = 0

    for video_info in tqdm(current_batch, desc="Processing"):
        result = process_single_video(video_info, metadata_df)
        if result:
            success_count += 1

    print(f"\nBatch complete: {success_count}/{len(current_batch)}")

    # Save processing position
    if current_batch:
        last_video = current_batch[-1]
        save_resume_position(last_video['group'], last_video['video_id'])
        print(f"Saved position: group={last_video['group']}, video_id={last_video['video_id']}")

    if len(remaining_videos) > success_count:
        sys.exit(1)  # Continue
    else:
        if os.path.exists(RESUME_FILE):
            os.remove(RESUME_FILE)
        sys.exit(0)  # Done

if __name__ == "__main__":
    main()
