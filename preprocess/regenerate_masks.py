#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate SAM3 masks for already-processed videos.
Processes only frames directories, generating masks for identified objects.
"""

import sys
import os
import json
import cv2
import numpy as np
import torch
import shutil
from pathlib import Path
from tqdm import tqdm

# Add SAM3 path
SAM3_PATH = "./sam3/sam3-main"
sys.path.insert(0, SAM3_PATH)

print("=" * 80)
print("SAM3 Mask Generation Script")
print("=" * 80)

# Check SAM3
try:
    from sam3.model_builder import build_sam3_video_predictor
    print("\n✓ SAM3success")
except ImportError as e:
    print(f"\n✗ SAM3 import failed: {e}")
    print("\nrunSAM3:")
    print(f"  cd {SAM3_PATH}")
    print("  pip install -e .")
    sys.exit(1)

# Initialize SAM3
print("\nloadSAM3...")
SAM3_CHECKPOINT = "./sam3/models/sam3.pt"

NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

try:
    # Do not pass gpus_to_use, let SAM3 use default current GPU
    # This avoids issues with multi-GPU distributed mode
    if NUM_GPUS > 0:
        print(f"Detected {NUM_GPUS}  GPUs, using default GPU")
    else:
        print("No GPU detected, using CPU")
    
    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CHECKPOINT
        # Do not pass gpus_to_use, use default single-GPU mode
    )
    print("✓ SAM3loadsuccess\n")
except Exception as e:
    print(f"✗ SAM3 model loading failed: {e}")
    sys.exit(1)

# Configuration
processed_dir = Path("./output")

# Statistics
total_videos = 0
processed_videos = 0
failed_videos = []

# Scan all processed videos
video_list = []
for group_dir in sorted(processed_dir.glob("group_*")):
    for video_dir in sorted(group_dir.iterdir()):
        if video_dir.is_dir() and (video_dir / "data.jsonl").exists():
            video_list.append(video_dir)

total_videos = len(video_list)
print(f"Found {total_videos}  processed videos\n")

if total_videos == 0:
    print("No videos found to process")
    sys.exit(0)

# Process each video
for video_dir in tqdm(video_list, desc="Generating masks"):
    try:
        data_file = video_dir / "data.jsonl"
        
        # Read data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        video_id = data['video_id']
        dynamic_obj = data.get('dynamic_object')
        static_obj = data.get('static_object')
        frames_dir = str(video_dir / "frames")
        
        # Check if processing is needed
        existing_dynamic_masks = [p for p in data.get('dynamic_mask_paths', []) if p and os.path.exists(p)]
        existing_static_masks = [p for p in data.get('static_mask_paths', []) if p and os.path.exists(p)]
        
        # Skip if mask already exists
        if existing_dynamic_masks or existing_static_masks:
            tqdm.write(f"Skipping {video_id}: already has mask")
            continue
        
        # Skip if no objects identified
        if (not dynamic_obj or dynamic_obj.lower() == 'null') and \
           (not static_obj or static_obj.lower() == 'null'):
            tqdm.write(f"Skipping {video_id}: no objects identified")
            continue
        
        dynamic_mask_paths = []
        static_mask_paths = []
        masked_frame_paths = []
        
        frame_files = sorted((video_dir / "frames").glob("*.jpg"))
        num_frames = len(frame_files)
        
        # Check if frames exceed 50
        if num_frames > 50:
            tqdm.write(f"Skipping {video_id}: too many frames ({num_frames} > 50)")
            # Clean up the directory
            if video_dir.exists():
                shutil.rmtree(video_dir)
            continue
        
        # Use the predictor (SAM3 will handle GPU management)
        tqdm.write(f"Processing {video_id}")
        
        # Store mask for each frame
        dynamic_masks_list = []
        static_masks_list = []
        
        # Create session for dynamic object
        dynamic_session_id = None
        if dynamic_obj and dynamic_obj.lower() != 'null':
            try:
                response = predictor.handle_request(
                    request=dict(type="start_session", resource_path=frames_dir)
                )
                dynamic_session_id = response.get("session_id")
                if not dynamic_session_id:
                    tqdm.write(f"  ✗ {video_id}: Dynamic object session creation failed")
            except Exception as e:
                tqdm.write(f"  ✗ {video_id}: Dynamic object session error - {e}")
        
        # Create session for static object
        static_session_id = None
        if static_obj and static_obj.lower() != 'null':
            try:
                response = predictor.handle_request(
                    request=dict(type="start_session", resource_path=frames_dir)
                )
                static_session_id = response.get("session_id")
                if not static_session_id:
                    tqdm.write(f"  ✗ {video_id}: Static object session creation failed")
            except Exception as e:
                tqdm.write(f"  ✗ {video_id}: Static object session error - {e}")
        
        # Generate mask for each frame
        for frame_idx in range(num_frames):
            # Process dynamic object current frame (with retry)
            if dynamic_session_id:
                mask_result = None
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries and mask_result is None:
                    try:
                        response = predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=dynamic_session_id,
                                frame_index=frame_idx,
                                text=dynamic_obj
                            )
                        )
                        
                        if "outputs" in response and "out_binary_masks" in response["outputs"]:
                            mask = response["outputs"]["out_binary_masks"]
                            # Convert to numpy if it's a tensor
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            
                            # Check if mask is empty (SAM3 did not detect object)
                            if mask.shape[0] == 0:
                                # empty mask, retrying
                                retry_count += 1
                                if retry_count < max_retries:
                                    if frame_idx == 0:  # only show retry info on first frame
                                        tqdm.write(f"  ⚠️ {video_id} dynamic object frame {frame_idx}: empty mask, retrying {retry_count}/{max_retries}")
                                    continue
                                else:
                                    # all retries failed, set to None
                                    mask_result = None
                            else:
                                # successfully obtained mask
                                if len(mask.shape) == 3:
                                    mask = mask[0]  # (N, H, W) -> (H, W)
                                mask_result = mask
                                if retry_count > 0 and frame_idx == 0:
                                    tqdm.write(f"  ✓ {video_id} dynamic object frame {frame_idx}: succeeded on attempt {retry_count+1}")
                        else:
                            retry_count += 1
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            if frame_idx == 0:
                                tqdm.write(f"  ✗ {video_id}: dynamicobjectframe {frame_idx}error - {e}")
                
                dynamic_masks_list.append(mask_result)
            
            # # Process static object current frame (with retry)
            if static_session_id:
                mask_result = None
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries and mask_result is None:
                    try:
                        response = predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=static_session_id,
                                frame_index=frame_idx,
                                text=static_obj
                            )
                        )
                        
                        if "outputs" in response and "out_binary_masks" in response["outputs"]:
                            mask = response["outputs"]["out_binary_masks"]
                            # Convert to numpy if it's a tensor
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            
                            # Check if mask is empty (SAM3 did not detect object)
                            if mask.shape[0] == 0:
                                # empty mask, retrying
                                retry_count += 1
                                if retry_count < max_retries:
                                    if frame_idx == 0:
                                        tqdm.write(f"  ⚠️ {video_id} static object frame {frame_idx}: empty mask, retrying {retry_count}/{max_retries}")
                                    continue
                                else:
                                    # all retries failed, set to None
                                    mask_result = None
                            else:
                                # successfully obtained mask
                                if len(mask.shape) == 3:
                                    mask = mask[0]  # (N, H, W) -> (H, W)
                                mask_result = mask
                                if retry_count > 0 and frame_idx == 0:
                                    tqdm.write(f"  ✓ {video_id} static object frame {frame_idx}: succeeded on attempt {retry_count+1}")
                        else:
                            retry_count += 1
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            if frame_idx == 0:
                                tqdm.write(f"  ✗ {video_id}: staticobjectframe {frame_idx}error - {e}")
                
                static_masks_list.append(mask_result)
        
        # # Count successfully generated masks
        dynamic_success = len([m for m in dynamic_masks_list if m is not None])
        static_success = len([m for m in static_masks_list if m is not None])
        
        if dynamic_session_id:
            tqdm.write(f"  ✓ {video_id}: dynamic object {dynamic_success}/{num_frames} frame masks generated ({dynamic_obj})")
        if static_session_id:
            tqdm.write(f"  ✓ {video_id}: static object {static_success}/{num_frames} frame masks generated ({static_obj})")
        
        # # Save masks and generate masked frames
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                dynamic_mask_paths.append("")
                static_mask_paths.append("")
                masked_frame_paths.append("")
                continue
            
            h, w = frame.shape[:2]
            frame_masked = frame.copy()
            
            # Processingdynamicmask
            if i < len(dynamic_masks_list) and dynamic_masks_list[i] is not None:
                mask = dynamic_masks_list[i]
                # Resize mask to match frame size
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                    mask = mask.astype(np.uint8) * 255
                else:
                    # Ensure mask is in 0-255 range
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
                
                # Save mask
                mask_path = video_dir / "masks_dynamic" / f"{frame_file.stem}_mask.npy"
                np.save(mask_path, mask)
                dynamic_mask_paths.append(str(mask_path))
                
                # Draw red mask overlay
                mask_3ch = np.stack([np.zeros_like(mask), np.zeros_like(mask), mask], axis=2)
                frame_masked = cv2.addWeighted(frame_masked, 1.0, mask_3ch, 0.5, 0)
            else:
                dynamic_mask_paths.append("")
            
            # Processingstaticmask
            if i < len(static_masks_list) and static_masks_list[i] is not None:
                mask = static_masks_list[i]
                # Resize mask to match frame size
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                    mask = mask.astype(np.uint8) * 255
                else:
                    # Ensure mask is in 0-255 range
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
                
                # Save mask
                mask_path = video_dir / "masks_static" / f"{frame_file.stem}_mask.npy"
                np.save(mask_path, mask)
                static_mask_paths.append(str(mask_path))
                
                # Draw blue mask overlay
                mask_3ch = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2)
                frame_masked = cv2.addWeighted(frame_masked, 1.0, mask_3ch, 0.5, 0)
            else:
                static_mask_paths.append("")
            
            # Save masked frame
            masked_frame_path = video_dir / "masked_frames" / frame_file.name
            cv2.imwrite(str(masked_frame_path), frame_masked)
            masked_frame_paths.append(str(masked_frame_path))
        
        # Update data.jsonl
        data['dynamic_mask_paths'] = dynamic_mask_paths
        data['static_mask_paths'] = static_mask_paths
        data['masked_frame_paths'] = masked_frame_paths
        
        with open(data_file, 'w') as f:
            json.dump(data, f)
            f.write('\n')
        
        processed_videos += 1
        
    except Exception as e:
        tqdm.write(f"✗ {video_dir.name}: Processing failed - {e}")
        failed_videos.append(video_dir.name)

# # Summary
print("\n" + "=" * 80)
print("ProcessingDone")
print("=" * 80)
print(f"Total videos: {total_videos}")
print(f"successProcessing: {processed_videos}")
print(f"Failed count: {len(failed_videos)}")

if failed_videos:
    print("\nFailed videos:")
    for vid in failed_videos[:10]:
        print(f"  - {vid}")
    if len(failed_videos) > 10:
        print(f"  ... and {len(failed_videos) - 10}  more")

print("\nDone!")

