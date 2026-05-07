#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import cv2
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI
import torch
from PIL import Image
import traceback
import time
import shutil
import pickle
from typing import List, Dict, Optional

# Add SAM3 to path
SAM3_PATH = "./sam3/sam3-main"
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

# Try to import SAM3
try:
    from sam3.model_builder import build_sam3_video_predictor
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM3 not available: {e}")
    SAM3_AVAILABLE = False
    build_sam3_video_predictor = None

# Configuration
BASE_VIDEO_DIR = "./data/SpatialVID/videos"
BASE_ANNO_DIR = "./data/SpatialVID/annotations"
METADATA_CSV = "./data/SpatialVID/SpatialVID_metadata.csv"
OUTPUT_BASE_DIR = "./output"
SAM3_CHECKPOINT = "./sam3/models/sam3.pt"
MAX_VIDEOS = 100000
MAX_FRAMES = 50
VIDEOS_PER_GPU_RESET = 20  # Reset after 20 videos

PROGRESS_FILE = os.path.join(OUTPUT_BASE_DIR, "processing_progress.pkl")
COMPLETED_FILE = os.path.join(OUTPUT_BASE_DIR, "completed_videos.txt")

# OpenAI Client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# Completed videos set
completed_videos = set()
completed_lock = threading.Lock()

# GPU configuration
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
NUM_WORKERS = NUM_GPUS if NUM_GPUS > 0 else 1  # Thread count equals GPU count

print(f"Detected {NUM_GPUS} GPUs, starting {NUM_WORKERS} worker threads")
print(f"Each GPU has one dedicated thread, each SAM3 model processes one request at a time")

# GPU locks - one lock per GPU, ensures one request at a time
gpu_locks = [threading.Lock() for _ in range(max(NUM_GPUS, 1))]

# Thread-local storage
thread_local = threading.local()

# Thread to GPU mapping
thread_gpu_map = {}
gpu_assignment_lock = threading.Lock()

# Video processing counter
thread_video_counter = {}
video_counter_lock = threading.Lock()


def load_progress():
    """Load processing progress."""
    global completed_videos
    
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, 'r') as f:
            completed_videos = set(line.strip() for line in f if line.strip())
        print(f"Loaded progress: {len(completed_videos)} completed videos")
    else:
        completed_videos = set()
        print("No progress file found, starting from scratch")


def save_completed_video(video_id: str):
    """Save completed video ID."""
    with completed_lock:
        completed_videos.add(video_id)
        with open(COMPLETED_FILE, 'a') as f:
            f.write(f"{video_id}\n")


def is_video_completed(video_id: str) -> bool:
    """Check if video is completed."""
    return video_id in completed_videos


def get_thread_gpu_id():
    """Get GPU ID assigned to current thread."""
    thread_id = threading.get_ident()
    
    with gpu_assignment_lock:
        if thread_id not in thread_gpu_map:
            # Assign GPU to new thread (round-robin)
            gpu_id = len(thread_gpu_map) % NUM_GPUS if NUM_GPUS > 0 else 0
            thread_gpu_map[thread_id] = gpu_id
            print(f"Thread {thread_id} assigned to GPU {gpu_id}")
        
        return thread_gpu_map[thread_id]


def reset_video_predictor():
    """Reset thread-local video predictor."""
    if hasattr(thread_local, 'predictor'):
        del thread_local.predictor
        
    if hasattr(thread_local, 'gpu_id'):
        gpu_id = thread_local.gpu_id
        if torch.cuda.is_available():
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        thread_id = threading.get_ident()
        print(f"  ♻️ Predictor reset for thread {thread_id} (GPU {gpu_id})")


def clear_all_gpu_memory():
    """Clear all GPU memory."""
    if torch.cuda.is_available():
        print("\n" + "!" * 80)
        print("⚠️  OOM DETECTED - Clearing all GPU memory...")
        print("!" * 80)
        
        # Clean up all thread-local predictors
        if hasattr(thread_local, 'predictor'):
            del thread_local.predictor
        
        # Clear all GPU caches
        for i in range(NUM_GPUS):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Reset counters
        thread_id = threading.get_ident()
        with video_counter_lock:
            if thread_id in thread_video_counter:
                thread_video_counter[thread_id] = 0
        
        import gc
        gc.collect()
        time.sleep(2)
        print("✓ GPU memory cleared\n")


def get_video_predictor():
    """Get thread-local video predictor (each thread owns one GPU)."""
    if not SAM3_AVAILABLE:
        return None
    
    thread_id = threading.get_ident()
    gpu_id = get_thread_gpu_id()
    
    # Check if periodic reset is needed
    with video_counter_lock:
        if thread_id not in thread_video_counter:
            thread_video_counter[thread_id] = 0
        
        if thread_video_counter[thread_id] > 0 and \
           thread_video_counter[thread_id] % VIDEOS_PER_GPU_RESET == 0:
            print(f"  Resetting predictor after {VIDEOS_PER_GPU_RESET} videos (Thread {thread_id}, GPU {gpu_id})")
            reset_video_predictor()
    
    if not hasattr(thread_local, 'predictor'):
        # Set current GPU
        torch.cuda.set_device(gpu_id)
        
        print(f"  Initializing SAM3 predictor for Thread {thread_id} on GPU {gpu_id}")
        
        thread_local.predictor = build_sam3_video_predictor(
            checkpoint_path=SAM3_CHECKPOINT
        )
        thread_local.gpu_id = gpu_id
        
        print(f"  ✓ SAM3 predictor initialized (Thread {thread_id}, GPU {gpu_id})")
    
    return thread_local.predictor


def get_all_videos():
    """Get all video paths organized by group"""
    video_list = []
    for group_dir in sorted(Path(BASE_VIDEO_DIR).iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        for video_file in sorted(group_dir.glob("*.mp4")):
            video_id = video_file.stem
            video_list.append({
                'group': group_name,
                'video_id': video_id,
                'video_path': str(video_file)
            })
    return video_list


def load_metadata(metadata_csv):
    """Load metadata CSV file"""
    try:
        df = pd.read_csv(metadata_csv)
        df['video_id'] = df['video path'].apply(lambda x: Path(x).stem)
        return df.set_index('video_id')
    except Exception as e:
        print(f"Warning: Failed to load metadata CSV: {e}")
        return None


def extract_frames_from_video(video_path, indexes, output_dir):
    """Extract frames from video based on indexes.txt"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_paths = []
    for frame_seq, frame_idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_filename = f"frame_{frame_seq:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def identify_objects_with_gpt(frame_paths, max_retry=3):
    """Use GPT-4 to identify moving and static objects"""
    sample_indices = np.linspace(0, len(frame_paths)-1, min(10, len(frame_paths)), dtype=int)
    sampled_frames = [frame_paths[i] for i in sample_indices]
    
    content = [{
        "type": "text",
        "text": "Analyze this video sequence and identify: 1) One object that is clearly moving/changing throughout the frames (dynamic object). 2) One object that remains static/unchanged throughout the frames (static object). Requirements: Both objects must be visible and prominent in ALL frames. Use SHORT and CLEAR labels (3-8 words max) with key features like color or shape, for example: 'a blue-handled knife', 'a red car', 'a person in white shirt'. If no clear dynamic or static object exists, respond with 'null' for that category. Respond in JSON format: {\"dynamic_object\": \"short clear label or null\", \"static_object\": \"short clear label or null\"}"
    }]
    
    for frame_path in sampled_frames:
        img_base64 = image_to_base64(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })
    
    for attempt in range(max_retry):
        try:
            result = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": content}],
                stream=False,
                extra_headers={"M-TraceId": str(int(time.time() * 1000))}
            )
            
            response_text = result.choices[0].message.content
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            obj_info = json.loads(response_text)
            return {
                'dynamic_object': obj_info.get('dynamic_object'),
                'static_object': obj_info.get('static_object')
            }
                
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"Error calling GPT API after {max_retry} retries: {e}")
            return {'dynamic_object': None, 'static_object': None}


def is_oom_error(exception):
    """Detect if error is OOM."""
    error_msg = str(exception).lower()
    return (
        'out of memory' in error_msg or
        'oom' in error_msg or
        'cuda out of memory' in error_msg or
        'cudnn error' in error_msg
    )


def extract_masks_with_sam3_sequential(frame_dir, dynamic_object, static_object, output_dir):
    """
    Extract masks using SAM3 - serial processing version.
    Each GPU processes one request at a time.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masked_frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_dynamic"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_static"), exist_ok=True)
    
    if not SAM3_AVAILABLE:
        return [], [], []
    
    # Get current thread GPU ID and lock
    gpu_id = get_thread_gpu_id()
    gpu_lock = gpu_locks[gpu_id]
    
    # Get predictor (outside lock since predictor is thread-local)
    predictor = get_video_predictor()
    if predictor is None:
        return [], [], []
    
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    num_frames = len(frame_files)
    
    masked_frame_paths = []
    dynamic_mask_paths = []
    static_mask_paths = []
    
    # Use GPU lock to ensure one request at a time
    with gpu_lock:
        thread_id = threading.get_ident()
        print(f"  [Thread {thread_id}, GPU {gpu_id}] Processing {num_frames} frames (LOCKED)")
        
        dynamic_session_id = None
        static_session_id = None
        
        try:
            # Create sessions
            if dynamic_object and dynamic_object.lower() != 'null':
                try:
                    print(f"  [GPU {gpu_id}] Creating dynamic session: {dynamic_object}")
                    response = predictor.handle_request(
                        request=dict(type="start_session", resource_path=frame_dir)
                    )
                    dynamic_session_id = response.get("session_id")
                    print(f"  [GPU {gpu_id}] ✓ Dynamic session: {dynamic_session_id}")
                except Exception as e:
                    if is_oom_error(e):
                        raise
                    print(f"  [GPU {gpu_id}] ✗ Dynamic session failed: {e}")
            
            if static_object and static_object.lower() != 'null':
                try:
                    print(f"  [GPU {gpu_id}] Creating static session: {static_object}")
                    response = predictor.handle_request(
                        request=dict(type="start_session", resource_path=frame_dir)
                    )
                    static_session_id = response.get("session_id")
                    print(f"  [GPU {gpu_id}] ✓ Static session: {static_session_id}")
                except Exception as e:
                    if is_oom_error(e):
                        raise
                    print(f"  [GPU {gpu_id}] ✗ Static session failed: {e}")
            
            # Process frames sequentially
            dynamic_masks_list = []
            static_masks_list = []
            
            for frame_idx in range(num_frames):
                # Process dynamic object
                if dynamic_session_id:
                    mask_result = None
                    try:
                        response = predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=dynamic_session_id,
                                frame_index=frame_idx,
                                text=dynamic_object,
                            )
                        )
                        if "outputs" in response and "out_binary_masks" in response["outputs"]:
                            mask = response["outputs"]["out_binary_masks"]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            
                            if mask.shape[0] > 0:
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                mask_result = mask
                    except Exception as e:
                        if is_oom_error(e):
                            raise
                        if frame_idx == 0:
                            print(f"  [GPU {gpu_id}] Frame {frame_idx} dynamic error: {e}")
                    
                    dynamic_masks_list.append(mask_result)
                
                # Process static object
                if static_session_id:
                    mask_result = None
                    try:
                        response = predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=static_session_id,
                                frame_index=frame_idx,
                                text=static_object,
                            )
                        )
                        if "outputs" in response and "out_binary_masks" in response["outputs"]:
                            mask = response["outputs"]["out_binary_masks"]
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            
                            if mask.shape[0] > 0:
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                mask_result = mask
                    except Exception as e:
                        if is_oom_error(e):
                            raise
                        if frame_idx == 0:
                            print(f"  [GPU {gpu_id}] Frame {frame_idx} static error: {e}")
                    
                    static_masks_list.append(mask_result)
            
            # Save masks and create masked frames
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    dynamic_mask_paths.append("")
                    static_mask_paths.append("")
                    masked_frame_paths.append("")
                    continue
                
                h, w = frame.shape[:2]
                frame_masked = frame.copy()
                
                # Apply dynamic mask (red)
                if i < len(dynamic_masks_list) and dynamic_masks_list[i] is not None:
                    mask = dynamic_masks_list[i]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                        mask = mask.astype(np.uint8) * 255
                    else:
                        if mask.max() <= 1.0:
                            mask = (mask * 255).astype(np.uint8)
                        else:
                            mask = mask.astype(np.uint8)
                    
                    mask_path = os.path.join(output_dir, "masks_dynamic", frame_file.replace('.jpg', '_mask.npy'))
                    np.save(mask_path, mask)
                    dynamic_mask_paths.append(mask_path)
                    
                    mask_3ch = np.stack([np.zeros_like(mask), np.zeros_like(mask), mask], axis=2)
                    frame_masked = cv2.addWeighted(frame_masked, 1.0, mask_3ch, 0.5, 0)
                else:
                    dynamic_mask_paths.append("")
                
                # Apply static mask (blue)
                if i < len(static_masks_list) and static_masks_list[i] is not None:
                    mask = static_masks_list[i]
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                        mask = mask.astype(np.uint8) * 255
                    else:
                        if mask.max() <= 1.0:
                            mask = (mask * 255).astype(np.uint8)
                        else:
                            mask = mask.astype(np.uint8)
                    
                    mask_path = os.path.join(output_dir, "masks_static", frame_file.replace('.jpg', '_mask.npy'))
                    np.save(mask_path, mask)
                    static_mask_paths.append(mask_path)
                    
                    mask_3ch = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2)
                    frame_masked = cv2.addWeighted(frame_masked, 1.0, mask_3ch, 0.5, 0)
                else:
                    static_mask_paths.append("")
                
                # Save masked frame
                masked_frame_path = os.path.join(output_dir, "masked_frames", frame_file)
                cv2.imwrite(masked_frame_path, frame_masked)
                masked_frame_paths.append(masked_frame_path)
        
        except Exception as e:
            if is_oom_error(e):
                raise
            print(f"  [GPU {gpu_id}] Error in SAM3 processing: {e}")
            traceback.print_exc()
        
        finally:
            # Clean up sessions - try multiple possible close methods
            if dynamic_session_id is not None:
                try:
                    # Try method 1: close_session
                    predictor.handle_request(
                        request=dict(type="close_session", session_id=dynamic_session_id)
                    )
                    print(f"  [GPU {gpu_id}] ✓ Dynamic session closed (close_session)")
                except Exception as e1:
                    try:
                        # Try method 2: stop_session
                        predictor.handle_request(
                            request=dict(type="stop_session", session_id=dynamic_session_id)
                        )
                        print(f"  [GPU {gpu_id}] ✓ Dynamic session closed (stop_session)")
                    except Exception as e2:
                        try:
                            # Try method 3: reset_session
                            predictor.handle_request(
                                request=dict(type="reset_session", session_id=dynamic_session_id)
                            )
                            print(f"  [GPU {gpu_id}] ✓ Dynamic session reset (reset_session)")
                        except Exception as e3:
                            # If all methods fail, print warning only
                            print(f"  [GPU {gpu_id}] ⚠️ Could not close dynamic session (tried close/stop/reset)")
                            # Skip detailed error to avoid excessive logs
            
            if static_session_id is not None:
                try:
                    # Try method 1: close_session
                    predictor.handle_request(
                        request=dict(type="close_session", session_id=static_session_id)
                    )
                    print(f"  [GPU {gpu_id}] ✓ Static session closed (close_session)")
                except Exception as e1:
                    try:
                        # Try method 2: stop_session
                        predictor.handle_request(
                            request=dict(type="stop_session", session_id=static_session_id)
                        )
                        print(f"  [GPU {gpu_id}] ✓ Static session closed (stop_session)")
                    except Exception as e2:
                        try:
                            # Try method 3: reset_session
                            predictor.handle_request(
                                request=dict(type="reset_session", session_id=static_session_id)
                            )
                            print(f"  [GPU {gpu_id}] ✓ Static session reset (reset_session)")
                        except Exception as e3:
                            # If all methods fail, print warning only
                            print(f"  [GPU {gpu_id}] ⚠️ Could not close static session (tried close/stop/reset)")
            
            # Clean up memory
            del dynamic_masks_list
            del static_masks_list
            
            # Force Python garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            
            print(f"  [Thread {thread_id}, GPU {gpu_id}] Processing complete (UNLOCKED)")
    
    return masked_frame_paths, dynamic_mask_paths, static_mask_paths


def process_single_video(video_info, metadata_df, max_oom_retries=3):
    """Process single video with OOM retry."""
    group = video_info['group']
    video_id = video_info['video_id']
    video_path = video_info['video_path']
    
    if is_video_completed(video_id):
        return None
    
    # Increment counter
    thread_id = threading.get_ident()
    with video_counter_lock:
        if thread_id not in thread_video_counter:
            thread_video_counter[thread_id] = 0
        thread_video_counter[thread_id] += 1
    
    # OOM retry loop
    for oom_attempt in range(max_oom_retries):
        try:
            return _process_single_video_inner(video_info, metadata_df, group, video_id, video_path)
        
        except Exception as e:
            if is_oom_error(e):
                print(f"\n⚠️  OOM ERROR for {video_id} (attempt {oom_attempt + 1}/{max_oom_retries})")
                clear_all_gpu_memory()
                reset_video_predictor()
                
                if oom_attempt < max_oom_retries - 1:
                    print(f"   → Retrying {video_id}...")
                    continue
                else:
                    print(f"   ✗ Failed after {max_oom_retries} retries, skipping {video_id}")
                    return None
            else:
                raise
    
    return None


def _process_single_video_inner(video_info, metadata_df, group, video_id, video_path):
    """Process single video - internal implementation."""
    output_dir = os.path.join(OUTPUT_BASE_DIR, "processed_all", group, video_id)
    
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # 1. Load annotations
        anno_dir = os.path.join(BASE_ANNO_DIR, group, video_id)
        if not os.path.exists(anno_dir):
            return None
        
        indexes_file = os.path.join(anno_dir, "indexes.txt")
        instructions_file = os.path.join(anno_dir, "instructions.json")
        caption_file = os.path.join(anno_dir, "caption.json")
        intrinsics_file = os.path.join(anno_dir, "intrinsics.npy")
        poses_file = os.path.join(anno_dir, "poses.npy")
        
        # Parse indexes
        indexes = []
        with open(indexes_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    indexes.append((int(parts[0]), int(parts[1])))
        
        # Load JSON files
        with open(instructions_file, 'r') as f:
            instructions = json.load(f)
        with open(caption_file, 'r') as f:
            caption = json.load(f)
        
        # Load numpy files
        intrinsics = np.load(intrinsics_file)
        poses = np.load(poses_file)
        
        # 2. Extract frames
        frames_dir = os.path.join(output_dir, "frames")
        frame_paths = extract_frames_from_video(video_path, indexes, frames_dir)
        
        if len(frame_paths) == 0:
            return None
        
        # Check frame limit
        if len(frame_paths) > MAX_FRAMES:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            return None
        
        # 3. Identify objects with GPT
        objects_info = identify_objects_with_gpt(frame_paths)
        dynamic_object = objects_info['dynamic_object']
        static_object = objects_info['static_object']
        
        # 4. Extract masks with SAM3 (serial processing)
        masked_frame_paths, dynamic_mask_paths, static_mask_paths = extract_masks_with_sam3_sequential(
            frames_dir, dynamic_object, static_object, output_dir
        )
        
        # 5. Prepare data for JSONL
        data_entry = {
            'video_id': video_id,
            'group': group,
            'video_path': video_path,
            'instructions': instructions,
            'caption': caption,
            'intrinsics': intrinsics.tolist(),
            'poses': poses.tolist(),
            'dynamic_object': dynamic_object,
            'static_object': static_object,
            'frame_paths': frame_paths,
            'masked_frame_paths': masked_frame_paths,
            'dynamic_mask_paths': dynamic_mask_paths,
            'static_mask_paths': static_mask_paths,
        }
        
        # Add metadata
        if metadata_df is not None and video_id in metadata_df.index:
            metadata_row = metadata_df.loc[video_id].to_dict()
            data_entry['metadata'] = metadata_row
        
        # 6. Write to JSONL
        jsonl_path = os.path.join(output_dir, "data.jsonl")
        with open(jsonl_path, 'w') as f:
            json.dump(data_entry, f)
            f.write('\n')
        
        # Save progress
        save_completed_video(video_id)
        
        # Clear GPU cache
        gpu_id = get_thread_gpu_id()
        if torch.cuda.is_available():
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        
        return video_id
        
    except Exception as e:
        if is_oom_error(e):
            raise
        
        print(f"Error processing {video_id}: {e}")
        traceback.print_exc()
        
        gpu_id = get_thread_gpu_id()
        if torch.cuda.is_available():
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        
        return None


def main():
    """Main processing pipeline"""
    print("=" * 80)
    print("SpatialVID Video Processing Pipeline (GPU-Exclusive Mode)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - GPUs: {NUM_GPUS}")
    print(f"  - Workers: {NUM_WORKERS} (1 worker per GPU)")
    print(f"  - Max frames per video: {MAX_FRAMES}")
    print(f"  - Videos per GPU reset: {VIDEOS_PER_GPU_RESET}")
    print("=" * 80)
    
    # Load progress
    print("\nLoading progress...")
    load_progress()
    
    # Load metadata
    print("\nLoading metadata...")
    metadata_df = load_metadata(METADATA_CSV)
    
    # Get all videos
    print("\nScanning for videos...")
    all_videos = get_all_videos()
    print(f"Found {len(all_videos)} videos")
    
    # Filter
    videos_to_process = [v for v in all_videos if not is_video_completed(v['video_id'])]
    videos_to_process = videos_to_process[:MAX_VIDEOS]
    
    print(f"Already completed: {len(completed_videos)} videos")
    print(f"Remaining to process: {len(videos_to_process)} videos")
    
    if len(videos_to_process) == 0:
        print("\nAll videos processed!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, "processed_all"), exist_ok=True)
    
    # Process videos
    print(f"\nStarting processing with {NUM_WORKERS} workers (1 per GPU)...")
    processed_count = 0
    failed_videos = []
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_single_video, video_info, metadata_df): video_info
            for video_info in videos_to_process
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            video_info = futures[future]
            try:
                result = future.result()
                if result:
                    processed_count += 1
                else:
                    failed_videos.append(video_info['video_id'])
            except Exception as e:
                print(f"\nUnexpected error for {video_info['video_id']}: {e}")
                failed_videos.append(video_info['video_id'])
    
    # Summary
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Total videos processed: {processed_count}/{len(videos_to_process)}")
    print(f"Failed videos: {len(failed_videos)}")
    print(f"Total completed: {len(completed_videos)}")
    
    if failed_videos:
        print("\nFailed video IDs:")
        for vid in failed_videos[:10]:
            print(f"  - {vid}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
    
    print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
    print(f"Progress file: {COMPLETED_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()