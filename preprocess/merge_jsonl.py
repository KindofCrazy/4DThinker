#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge all processed data.jsonl files.
Combines per-video data.jsonl into a single merged jsonl file.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Configuration
PROCESSED_DIR = "./output"
OUTPUT_JSONL = "./output/merged_data.jsonl"
STATS_OUTPUT = "./output/merge_stats.json"

print("=" * 80)
print("Merging data.jsonl files")
print("=" * 80)
print(f"Source directory: {PROCESSED_DIR}")
print(f"Output file: {OUTPUT_JSONL}")
print()

# Statistics
stats = {
    'total_videos': 0,
    'successful_videos': 0,
    'failed_videos': 0,
    'videos_with_dynamic_object': 0,
    'videos_with_static_object': 0,
    'videos_with_both_objects': 0,
    'videos_with_no_objects': 0,
    'total_frames': 0,
    'groups': {},
    'failed_files': [],
    'timestamp': datetime.now().isoformat()
}

# Scan all data.jsonl files
print("Scanning data.jsonl files...")
processed_dir = Path(PROCESSED_DIR)
jsonl_files = []

for group_dir in sorted(processed_dir.glob("group_*")):
    if not group_dir.is_dir():
        continue
    group_name = group_dir.name
    stats['groups'][group_name] = 0
    
    for video_dir in sorted(group_dir.iterdir()):
        if video_dir.is_dir():
            data_file = video_dir / "data.jsonl"
            if data_file.exists():
                jsonl_files.append((group_name, data_file))
                stats['total_videos'] += 1

print(f"Found {stats['total_videos']}  data.jsonl files")
print()

if stats['total_videos'] == 0:
    print("No data.jsonl files found!")
    exit(0)

# Merge all jsonl files
print("Merging data.jsonl files...")
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
    for group_name, jsonl_path in tqdm(jsonl_files, desc="Merging"):
        try:
            # Read jsonl file
            with open(jsonl_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
            
            # Write to merged file
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')
            
            # Update statistics
            stats['successful_videos'] += 1
            stats['groups'][group_name] += 1
            
            # Count frames
            frame_count = len(data.get('frame_paths', []))
            stats['total_frames'] += frame_count
            
            # Count object identification
            dynamic_obj = data.get('dynamic_object')
            static_obj = data.get('static_object')
            
            has_dynamic = dynamic_obj and dynamic_obj.lower() != 'null'
            has_static = static_obj and static_obj.lower() != 'null'
            
            if has_dynamic:
                stats['videos_with_dynamic_object'] += 1
            if has_static:
                stats['videos_with_static_object'] += 1
            
            if has_dynamic and has_static:
                stats['videos_with_both_objects'] += 1
            elif not has_dynamic and not has_static:
                stats['videos_with_no_objects'] += 1
                
        except Exception as e:
            stats['failed_videos'] += 1
            stats['failed_files'].append({
                'file': str(jsonl_path),
                'error': str(e)
            })
            tqdm.write(f"✗ Processing failed: {jsonl_path} - {e}")

# Save statistics
print("\nSaving statistics...")
with open(STATS_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# Print summary
print("\n" + "=" * 80)
print("Merge complete")
print("=" * 80)
print(f"Total videos: {stats['total_videos']}")
print(f"Successfully merged: {stats['successful_videos']}")
print(f"Failed: {stats['failed_videos']}")
print()
print(f"Total frames: {stats['total_frames']}")
print(f"Average frames: {stats['total_frames'] / max(stats['successful_videos'], 1):.1f}")
print()
print("Object identification statistics:")
print(f"  - Has dynamic object: {stats['videos_with_dynamic_object']} ({stats['videos_with_dynamic_object']/max(stats['successful_videos'],1)*100:.1f}%)")
print(f"  - Has static object: {stats['videos_with_static_object']} ({stats['videos_with_static_object']/max(stats['successful_videos'],1)*100:.1f}%)")
print(f"  - Has both: {stats['videos_with_both_objects']} ({stats['videos_with_both_objects']/max(stats['successful_videos'],1)*100:.1f}%)")
print(f"  - Has neither: {stats['videos_with_no_objects']} ({stats['videos_with_no_objects']/max(stats['successful_videos'],1)*100:.1f}%)")
print()
print("Videos per group:")
for group_name in sorted(stats['groups'].keys()):
    count = stats['groups'][group_name]
    print(f"  - {group_name}: {count}")
print()

if stats['failed_videos'] > 0:
    print("Failed files:")
    for failed in stats['failed_files'][:10]:
        print(f"  - {failed['file']}: {failed['error']}")
    if len(stats['failed_files']) > 10:
        print(f"  ... and {len(stats['failed_files']) - 10}  more")
    print()

print(f"Merged file saved to: {OUTPUT_JSONL}")
print(f"Statistics saved to: {STATS_OUTPUT}")
print("\nDone!")

