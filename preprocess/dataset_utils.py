#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Efficient dataset utility classes.
Provides fast loading, indexing, and batch processing functionality.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import cv2
from collections import defaultdict


class SpatialVIDDataset:
    """
    Efficient loader for SpatialVID dataset.
    Supports indexing, filtering, and batch loading.
    """

    def __init__(self, jsonl_path: str, cache_index: bool = True):
        """
        Initialize the dataset.

        Args:
            jsonl_path: Path to the merged jsonl file
            cache_index: Whether to cache the index to disk
        """
        self.jsonl_path = jsonl_path
        self.cache_index = cache_index
        self.index_path = jsonl_path.replace('.jsonl', '_index.pkl')

        # Index for fast data access
        self.video_index = []  # [(video_id, file_offset, data_length)]
        self.group_index = defaultdict(list)  # group -> [video_indices]
        self.object_index = defaultdict(list)  # object_name -> [video_indices]

        # Statistics
        self.stats = {
            'total_videos': 0,
            'total_frames': 0,
            'avg_frames': 0.0
        }

        # Build index
        self._build_index()

    def _build_index(self):
        """Build or load index."""
        # Try loading cached index
        if self.cache_index and os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    cached = pickle.load(f)
                    self.video_index = cached['video_index']
                    self.group_index = cached['group_index']
                    self.object_index = cached['object_index']
                    self.stats = cached['stats']
                print(f"Loaded cached index: {len(self.video_index)} videos")
                return
            except Exception as e:
                print(f"Warning: Failed to load index cache: {e}, rebuilding")

        # Build new index
        print("Building dataset index...")
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            offset = 0
            idx = 0
            total_frames = 0

            while True:
                line = f.readline()
                if not line:
                    break

                length = len(line.encode('utf-8'))

                try:
                    data = json.loads(line)
                    video_id = data['video_id']
                    group = data['group']

                    # Add to index
                    self.video_index.append((video_id, offset, length))
                    self.group_index[group].append(idx)

                    # Index objects
                    dynamic_obj = data.get('dynamic_object')
                    static_obj = data.get('static_object')
                    if dynamic_obj and dynamic_obj.lower() != 'null':
                        self.object_index[dynamic_obj].append(idx)
                    if static_obj and static_obj.lower() != 'null':
                        self.object_index[static_obj].append(idx)

                    # Statistics
                    total_frames += len(data.get('frame_paths', []))
                    idx += 1

                except Exception as e:
                    print(f"Warning: Parse failed at offset={offset}: {e}")

                offset += length

        self.stats['total_videos'] = len(self.video_index)
        self.stats['total_frames'] = total_frames
        self.stats['avg_frames'] = total_frames / max(len(self.video_index), 1)

        print(f"Index built: {self.stats['total_videos']} videos")

        # Save index to cache
        if self.cache_index:
            try:
                with open(self.index_path, 'wb') as f:
                    pickle.dump({
                        'video_index': self.video_index,
                        'group_index': self.group_index,
                        'object_index': self.object_index,
                        'stats': self.stats
                    }, f)
                print(f"Index cached to: {self.index_path}")
            except Exception as e:
                print(f"Warning: Failed to save index cache: {e}")

    def __len__(self):
        """Return dataset size."""
        return len(self.video_index)

    def __getitem__(self, idx: int) -> Dict:
        """Get data at specified index."""
        if idx < 0 or idx >= len(self.video_index):
            raise IndexError(f"Index {idx} out of range [0, {len(self.video_index)})")

        video_id, offset, length = self.video_index[idx]

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.read(length)
            return json.loads(line)

    def get_by_video_id(self, video_id: str) -> Optional[Dict]:
        """Get data by video_id."""
        for idx, (vid, _, _) in enumerate(self.video_index):
            if vid == video_id:
                return self[idx]
        return None

    def get_by_group(self, group: str) -> List[Dict]:
        """Get all videos in a specified group."""
        indices = self.group_index.get(group, [])
        return [self[idx] for idx in indices]

    def get_by_object(self, object_name: str) -> List[Dict]:
        """Get all videos containing a specified object."""
        indices = self.object_index.get(object_name, [])
        return [self[idx] for idx in indices]

    def filter(self,
               has_dynamic: bool = None,
               has_static: bool = None,
               min_frames: int = None,
               max_frames: int = None,
               group: str = None) -> List[int]:
        """
        Filter dataset.

        Returns:
            List of video indices matching the criteria.
        """
        result = []

        for idx in range(len(self.video_index)):
            data = self[idx]

            # Check dynamic object
            if has_dynamic is not None:
                dynamic_obj = data.get('dynamic_object')
                has_dyn = dynamic_obj and dynamic_obj.lower() != 'null'
                if has_dyn != has_dynamic:
                    continue

            # Check static object
            if has_static is not None:
                static_obj = data.get('static_object')
                has_sta = static_obj and static_obj.lower() != 'null'
                if has_sta != has_static:
                    continue

            # Check frame count
            frame_count = len(data.get('frame_paths', []))
            if min_frames is not None and frame_count < min_frames:
                continue
            if max_frames is not None and frame_count > max_frames:
                continue

            # Check group
            if group is not None and data.get('group') != group:
                continue

            result.append(idx)

        return result

    def iter_batches(self, batch_size: int = 32,
                     filter_fn=None,
                     load_frames: bool = False) -> Iterator[List[Dict]]:
        """
        Iterate dataset in batches.

        Args:
            batch_size: Batch size
            filter_fn: Filter function
            load_frames: Whether to load image frames
        """
        batch = []

        for idx in range(len(self.video_index)):
            data = self[idx]

            # Apply filter
            if filter_fn and not filter_fn(data):
                continue

            # Load frames if needed
            if load_frames:
                data['frames'] = self._load_frames(data['frame_paths'])

            batch.append(data)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Return last batch
        if batch:
            yield batch

    def _load_frames(self, frame_paths: List[str]) -> List[np.ndarray]:
        """Load image frames."""
        frames = []
        for path in frame_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    frames.append(img)
        return frames

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            **self.stats,
            'groups': {k: len(v) for k, v in self.group_index.items()},
            'unique_objects': len(self.object_index),
            'top_objects': sorted(
                [(k, len(v)) for k, v in self.object_index.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]
        }


class FrameLoader:
    """Efficient frame loader with batch loading and caching."""

    def __init__(self, cache_size: int = 100):
        """
        Args:
            cache_size: LRU cache size
        """
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []

    def load_frame(self, path: str) -> Optional[np.ndarray]:
        """Load a single frame."""
        # Check cache
        if path in self.cache:
            return self.cache[path]

        # Load from disk
        if not os.path.exists(path):
            return None

        img = cv2.imread(path)
        if img is None:
            return None

        # Add to cache
        self._add_to_cache(path, img)

        return img

    def load_frames_batch(self, paths: List[str]) -> List[np.ndarray]:
        """Load frames in batch."""
        return [self.load_frame(p) for p in paths if p]

    def _add_to_cache(self, path: str, img: np.ndarray):
        """Add to LRU cache."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        self.cache[path] = img
        self.cache_order.append(path)

    def clear_cache(self):
        """Clear cache."""
        self.cache.clear()
        self.cache_order.clear()


def create_dataset_splits(jsonl_path: str,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1,
                          seed: int = 42) -> Dict[str, str]:
    """
    Create train/val/test splits.

    Returns:
        {'train': path, 'val': path, 'test': path}
    """
    import random
    random.seed(seed)

    # Load all data
    videos = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            videos.append(line)

    # Shuffle
    random.shuffle(videos)

    # Split
    total = len(videos)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_videos = videos[:train_end]
    val_videos = videos[train_end:val_end]
    test_videos = videos[val_end:]

    # Save
    base_path = jsonl_path.replace('.jsonl', '')
    splits = {}

    for split_name, split_videos in [
        ('train', train_videos),
        ('val', val_videos),
        ('test', test_videos)
    ]:
        split_path = f"{base_path}_{split_name}.jsonl"
        with open(split_path, 'w', encoding='utf-8') as f:
            f.writelines(split_videos)
        splits[split_name] = split_path
        print(f"{split_name}: {len(split_videos)} videos -> {split_path}")

    return splits


# Usage example
if __name__ == "__main__":
    dataset = SpatialVIDDataset("merged_data.jsonl")

    print("\nDataset statistics:")
    stats = dataset.get_statistics()
    print(f"Total videos: {stats['total_videos']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Average frames: {stats['avg_frames']:.1f}")
    print(f"\nVideos per group:")
    for group, count in stats['groups'].items():
        print(f"  {group}: {count}")

    print(f"\nTop 20 most common objects:")
    for obj, count in stats['top_objects']:
        print(f"  {obj}: {count}")

    # Filter example
    print("\n\nFilter example:")
    filtered = dataset.filter(has_dynamic=True, has_static=True, min_frames=5)
    print(f"Videos with both dynamic and static objects, >=5 frames: {len(filtered)}")

    # Batch iteration example
    print("\nBatch iteration example:")
    for i, batch in enumerate(dataset.iter_batches(batch_size=10)):
        print(f"Batch {i}: {len(batch)} videos")
        if i >= 2:
            break
