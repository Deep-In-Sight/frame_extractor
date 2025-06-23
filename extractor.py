import os
import subprocess
import cv2
import threading
import time
from tqdm import tqdm
from typing import Dict, List
import re
from pathlib import Path
from datetime import datetime
import pysrt

class ProgressMonitor:
    def __init__(self, out_folder, total_frames, desc=None, interval=0.5):
        self.out_folder = out_folder
        self.total_frames = total_frames
        self.desc = desc
        self.interval = interval
        self._stop_flag = threading.Event()
        self._thread = None
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm(total=self.total_frames, desc=self.desc, leave=False)
        self._thread = threading.Thread(target=self._update_progress)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_flag.set()
        self._thread.join()
        self.pbar.n = self.total_frames
        self.pbar.refresh()
        self.pbar.close()

    def _update_progress(self):
        last_count = 0
        while not self._stop_flag.is_set():
            jpg_count = len([f for f in os.listdir(self.out_folder) if f.endswith('.jpg')])
            self.pbar.update(jpg_count - last_count)
            last_count = jpg_count
            if last_count >= self.total_frames:
                break
            time.sleep(self.interval)

def parse_srt_metadata(srt_path: str) -> Dict[int, Dict]:
    """Parse frame metadata from SRT file using pysrt"""
    if not os.path.exists(srt_path):
        return {}
        
    subs = pysrt.open(srt_path)
    frame_metas = {}
    
    for sub in subs:
        meta = {
            "frame_number": int(sub.index),
            "start_time": str(sub.start.to_time()).replace(',', '.'),
            "end_time": str(sub.end.to_time()).replace(',', '.')
        }
        
        # Parse the text content
        lines = sub.text.strip().split('\n')
        if not lines:
            continue
            
        # Remove font tags if present
        text = lines[0]
        if text.startswith('<font'):
            text = re.search(r'<font.*?>(.*?)</font>', text)
            if text:
                lines = text.group(1).strip().split('\n')
            
        # Parse frame count and diff time
        frame_info = re.search(r'FrameCnt: (\d+), DiffTime: (\d+)ms', lines[0])
        if frame_info:
            meta['frame_count'] = int(frame_info.group(1))
            meta['diff_time_ms'] = int(frame_info.group(2))
        
        # Parse timestamp if present
        if len(lines) > 1:
            try:
                meta['timestamp'] = datetime.strptime(
                    lines[1].strip(), 
                    '%Y-%m-%d %H:%M:%S.%f'
                ).isoformat()
            except ValueError:
                pass
        
        # Parse bracketed metadata if present
        if len(lines) > 2:
            brackets = re.findall(r'\[(.*?)\]', lines[2])
            for item in brackets:
                try:
                    key, value = item.split(':')
                    key = key.strip()
                    value = value.strip()
                    # Convert numeric values
                    try:
                        if '/' in value:  # Handle fractions (e.g., shutter speed)
                            num, denom = value.split('/')
                            value = float(num) / float(denom)
                        elif '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    meta[key] = value
                except ValueError:
                    continue
                    
        frame_metas[meta['frame_number']] = meta
        
    return frame_metas

def extract_video_metadata(video_path: str) -> Dict:
    """Extract video metadata including SRT metadata if available"""
    # Use OpenCV to get FPS and duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "frame_metas": {}
    }
    
    # Look for associated SRT file
    srt_path = str(Path(video_path).with_suffix('.srt'))
    if os.path.exists(srt_path):
        metadata["frame_metas"] = parse_srt_metadata(srt_path)
        metadata["has_srt"] = True
    else:
        metadata["has_srt"] = False
    
    return metadata

def extract_frames(video_paths: List[str], output_dir: str, verbose: bool = False, skip_ffmpeg: bool = False) -> Dict:
    metadata = {}
    # Parse metadata for all videos first
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        meta = extract_video_metadata(video_path)
        metadata[video_name] = meta

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_folder = os.path.join(output_dir, "extracted", video_name)
        os.makedirs(out_folder, exist_ok=True)
        out_pattern = os.path.join(out_folder, f"{video_name}_%06d.jpg")
        total_frames = metadata[video_name]["frame_count"]

        if not skip_ffmpeg:
            # ffmpeg command
            cmd = [
                "ffmpeg", '-hwaccel', 'auto', "-i", video_path, "-q:v", "1", out_pattern
            ]
            if not verbose:
                cmd.extend(["-hide_banner", "-loglevel", "error"])
            with ProgressMonitor(out_folder, total_frames, desc=f"Extracting {video_name}"):
                subprocess.run(cmd, check=True)

    return metadata 