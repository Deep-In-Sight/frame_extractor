import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from threading import Lock
import matplotlib.pyplot as plt
import math
from pathlib import Path
from typing import Dict, List, Tuple

class ImageSelector:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def process_group(self, group, src_dir, dst_dir):
        """Select the sharpest frame from a group, copy it to dst_dir, return the destination path and variance values."""
        sharpest = None
        max_var = -100000000
        group_vars = {}
        for fname in group:
            img_path = Path(src_dir) / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            var = np.var(cv2.Laplacian(img, cv2.CV_64F))
            group_vars[fname] = var
            if var > max_var:
                max_var = var
                sharpest = fname
        if sharpest:
            src = Path(src_dir) / sharpest
            dst = Path(dst_dir) / sharpest
            shutil.copy2(src, dst)
            return str(dst), group_vars
        return None, group_vars

    def process_video(self, video_name, input_dir, output_dir, target_fps, meta):
        """Process a single video: setup dirs, list frames, group, select sharpest, copy, return selected paths and variance values."""
        src_dir = Path(input_dir) / video_name
        dst_dir = Path(output_dir) / "selected" / video_name
        dst_dir.mkdir(exist_ok=True, parents=True)
        
        frame_files = sorted([f.name for f in src_dir.glob("*.jpg")])
        duration = meta["duration"]
        total_frames = len(frame_files)
        if duration == 0 or total_frames == 0:
            if self.verbose:
                print(f"Skipping {video_name}: duration or frame count is zero.")
            return [], {}
        num_groups = math.ceil(duration * target_fps)
        if num_groups == 0:
            num_groups = 1
        group_size = total_frames / num_groups
        groups = [
            frame_files[int(i*group_size):int((i+1)*group_size)] if i < num_groups - 1 else frame_files[int(i*group_size):]
            for i in range(num_groups)
        ]
        print(f"total frames {total_frames}, duration {duration}, num_groups {num_groups}, group_size {group_size}")
        
        # Process groups in parallel with tqdm's thread_map
        all_vars = {}
        
        def process_group_wrapper(group):
            if not group:
                return None, {}
            result, group_vars = self.process_group(group, str(src_dir), str(dst_dir))
            return result, group_vars
            
        results = thread_map(process_group_wrapper, groups, 
                           desc=f"Selecting for {video_name}",
                           max_workers=None)
        
        # Unpack results
        selected = []
        for result, group_vars in results:
            if result:
                selected.append(result)
            all_vars.update(group_vars)
            
        if self.verbose:
            print(f"Selected {len(selected)} frames for {video_name}")
        return selected, all_vars

    def select(self, input_dir, output_dir, target_fps, metadata, synchronized: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
        """
        Select frames and return selected paths and their metadata.
        
        Args:
            input_dir: Input directory containing extracted frames
            output_dir: Output directory for selected frames  
            target_fps: Target FPS for frame selection
            metadata: Video metadata dictionary
            synchronized: If True, use frame indices from first video for all videos
        
        Returns:
            Tuple containing:
            - Dict[str, List[str]]: Mapping of video names to lists of selected frame paths
            - Dict[str, Dict]: Mapping of video names to selected frame metadata
        """
        selected_paths = {}
        selected_metadata = {}
        
        # Sort videos by frame count
        sorted_videos = sorted(metadata.items(), key=lambda x: x[1].get("frame_count", 0))
        
        reference_frame_indices = None
        
        for video_name, meta in sorted_videos:
            if synchronized and reference_frame_indices is not None:
                # Use reference frame indices from first video
                selected, frame_vars = self.process_video_synchronized(
                    video_name, input_dir, output_dir, reference_frame_indices, meta
                )
            else:
                # Normal processing
                selected, frame_vars = self.process_video(video_name, input_dir, output_dir, target_fps, meta)
                
                # If synchronized and this is the first video, extract frame indices as reference
                if synchronized and reference_frame_indices is None:
                    reference_frame_indices = self.extract_frame_indices(selected)
            
            selected_paths[video_name] = selected
            
            # Extract metadata for selected frames if available
            if "frame_metas" in meta and meta["frame_metas"]:
                # Create metadata for selected frames
                selected_frame_metas = {}
                
                for path in selected:
                    # Extract frame number from filename
                    frame_num = int(Path(path).stem.split('_')[-1])
                    if frame_num in meta["frame_metas"]:
                        selected_frame_metas[frame_num] = meta["frame_metas"][frame_num]
                
                if selected_frame_metas:
                    selected_metadata[video_name] = {
                        "frame_metas": selected_frame_metas
                    }
                
        return selected_paths, selected_metadata

    def extract_frame_indices(self, selected_paths):
        """Extract frame indices from selected file paths."""
        frame_indices = []
        for path in selected_paths:
            # Extract frame number from filename (e.g., vid0_000123.jpg -> 123)
            frame_num = int(Path(path).stem.split('_')[-1])
            frame_indices.append(frame_num)
        return sorted(frame_indices)

    def process_video_synchronized(self, video_name, input_dir, output_dir, reference_frame_indices, meta):
        """Process a video using synchronized frame indices without quality assessment."""
        src_dir = Path(input_dir) / video_name
        dst_dir = Path(output_dir) / "selected" / video_name
        dst_dir.mkdir(exist_ok=True, parents=True)
        
        frame_files = sorted([f.name for f in src_dir.glob("*.jpg")])
        total_frames = len(frame_files)
        
        if total_frames == 0:
            if self.verbose:
                print(f"Skipping {video_name}: no frames found.")
            return [], {}
        
        selected = []
        frame_vars = {}
        
        # Copy frames based on reference indices
        for frame_idx in reference_frame_indices:
            if frame_idx <= total_frames:
                # Find the corresponding frame file
                frame_filename = f"{video_name}_{frame_idx:06d}.jpg"
                src_path = src_dir / frame_filename
                dst_path = dst_dir / frame_filename
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    selected.append(str(dst_path))
                    # Set a dummy variance value for consistency
                    frame_vars[frame_filename] = 0.0
                elif self.verbose:
                    print(f"Warning: Frame {frame_filename} not found in {video_name}")
        
        if self.verbose:
            print(f"Synchronized selection: {len(selected)} frames for {video_name}")
        
        return selected, frame_vars
