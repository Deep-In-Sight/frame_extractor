import os
import pytest
from pathlib import Path
import cv2
import pysrt
from datetime import datetime, timedelta
from extractor import parse_srt_metadata, extract_video_metadata
import numpy as np

def create_test_srt(path: str, num_frames: int = 60):
    """Create a test SRT file with frame metadata"""
    subs = []
    base_time = datetime.now()
    
    for i in range(num_frames):
        start_time = base_time + timedelta(milliseconds=i*33.33)  # ~30fps
        end_time = start_time + timedelta(milliseconds=33.33)
        
        sub = pysrt.SubRipItem(
            index=i,  # This becomes frame_number
            start=pysrt.SubRipTime.from_time(start_time.time()),
            end=pysrt.SubRipTime.from_time(end_time.time()),
            text=f"FrameCnt: {i}, DiffTime: 33ms\n{start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n[ISO:100] [Shutter:1/60] [GPS:12.34,56.78]"
        )
        subs.append(sub)
    
    with open(path, 'w', encoding='utf-8') as f:
        for sub in subs:
            f.write(str(sub) + '\n\n')

def test_parse_srt_metadata(tmp_path):
    # Create test SRT file
    srt_path = tmp_path / "test.srt"
    create_test_srt(str(srt_path))
    
    # Parse metadata
    frame_metas = parse_srt_metadata(str(srt_path))
    
    # Verify it's a dictionary
    assert isinstance(frame_metas, dict)
    
    # Check frame numbers as keys
    assert set(frame_metas.keys()) == set(range(60))
    
    # Check metadata structure for a sample frame
    frame_10 = frame_metas[10]
    assert frame_10['frame_number'] == 10
    assert 'start_time' in frame_10
    assert 'end_time' in frame_10
    assert frame_10['ISO'] == 100
    assert frame_10['Shutter'] == 1/60
    assert isinstance(frame_10['timestamp'], str)

def test_extract_video_metadata(tmp_path):
    # Create a test video file
    video_path = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (640, 480)
    )
    
    # Write 60 black frames
    for _ in range(60):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    
    # Create corresponding SRT file
    srt_path = video_path.with_suffix('.srt')
    create_test_srt(str(srt_path))
    
    # Extract metadata
    metadata = extract_video_metadata(str(video_path))
    
    # Verify structure
    assert isinstance(metadata, dict)
    assert metadata['fps'] == 30.0
    assert metadata['frame_count'] == 60
    assert metadata['duration'] == 2.0  # 60 frames at 30fps = 2 seconds
    assert metadata['has_srt'] is True
    
    # Verify frame_metas is a dictionary
    assert isinstance(metadata['frame_metas'], dict)
    assert len(metadata['frame_metas']) == 60
    
    # Check a sample frame's metadata
    frame_15 = metadata['frame_metas'][15]
    assert frame_15['frame_number'] == 15
    assert frame_15['ISO'] == 100
    assert frame_15['Shutter'] == 1/60

def test_missing_srt():
    # Test with non-existent SRT file
    frame_metas = parse_srt_metadata("nonexistent.srt")
    assert isinstance(frame_metas, dict)
    assert len(frame_metas) == 0 