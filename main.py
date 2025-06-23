# main.py

import tyro
from typing import List
from dataclasses import dataclass, field
import os
import json
import cv2
import shutil
from pathlib import Path
from object_masker import ObjectMasker
from selector import ImageSelector
from extractor import extract_frames
from veggie_masker import VeggieMasker
from util import flip_images_vertical, combine_masks

def copy_dir(src, dst, files=True):
    """
    Copy directory contents with flexible destination structure.
    
    Args:
        src (str/Path): Source directory path
        dst (str/Path): Destination directory path  
        files (bool): If True, copy src/* to dst/ (flat structure)
                     If False, copy src to dst/src_name (preserve directory structure)
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists() or not src.is_dir():
        return
    
    dst.mkdir(exist_ok=True, parents=True)
    if files:
        # get list of all files in the src tree
        files = list(src.rglob('*'))
        for file in files:
            if file.is_file():
                shutil.copy2(file, dst / file.name)
    else:
        # Copy src to dst/src_name (preserve directory structure)
        shutil.copytree(src, dst, dirs_exist_ok=True)

@dataclass
class Args:
    video_names: List[str]   # List of video paths to process (required)
    target_fps: int = 2      # Target number of frames per second to keep
    skip_extraction: bool = False       # Skip frame extraction, use existing frames
    skip_selection: bool = False        # Skip frame selection, use existing frames
    vflip: bool = False                 # Vertically flip selected images
    synchronized: bool = False          # Use synchronized frame selection across videos
    skip_objects_detection: bool = False  # Skip objects detection and masking
    detect_class: List[str] = field(default_factory=lambda: ['person', 'car', 'bus', 'truck', 'motorcycle'])
    skip_vegetation: bool = False        # Skip vegetation mask generation
    vegetation_area: int = 800          # Minimum area for vegetation patches
    keep_temp_files: bool = False        # Keep the temporary files
    verbose: bool = False               # Verbose logging
    merge: bool = True                  # Merge all images/masks into flat structure (no subfolders)

def pipeline(args: Args):
    # Validate required args
    if not args.video_names:
        raise ValueError("--video-names is required.")
    if args.verbose:
        print("Parsed arguments:")
        print(args)

    # Setup output directory (we're already in video directory)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract just the filenames from the full paths for processing
    video_paths = [os.path.basename(path) for path in args.video_names]

    # Frame extraction step and metadata
    metadata = extract_frames(video_paths, str(output_dir), args.verbose, skip_ffmpeg=args.skip_extraction)
    if args.skip_extraction and args.verbose:
        print("Skipping frame extraction. Using existing frames in extracted/ directory.")
    if args.verbose:
        print("\nVideo metadata:")
        print(json.dumps(metadata, indent=2))

    # Check for static masks and prepare static mask directory
    static_root = output_dir / "static"
    static_masks_found = {}
    
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = Path(video_path).parent
        static_mask_path = video_dir / f"{video_name}_mask.png"
        
        if static_mask_path.exists():
            static_masks_found[video_name] = static_mask_path
            if args.verbose:
                print(f"Found static mask for {video_name}: {static_mask_path}")

    # Image quality assessment and frame selection
    if not args.skip_selection:
        extracted_dir = output_dir / "extracted"
        selected_paths, selected_metadata = ImageSelector(verbose=args.verbose).select(str(extracted_dir), str(output_dir), args.target_fps, metadata, synchronized=args.synchronized)
        
        # Save selected metadata
        metadata_path = output_dir / "selected_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(selected_metadata, f, indent=2)
        if args.verbose:
            print(f"\nSaved selected frames metadata to {metadata_path}")
            print("\nSelected sharpest frames:")

    # Clone static masks for selected frames
    if static_masks_found:
        if args.verbose:
            print("\nCloning static masks for selected frames...")
        
        selected_root = output_dir / "selected"
        static_root.mkdir(exist_ok=True, parents=True)
        
        for video_name, static_mask_path in static_masks_found.items():
            video_dir = selected_root / video_name
            static_video_dir = static_root / video_name
            static_video_dir.mkdir(exist_ok=True, parents=True)
            
            if video_dir.is_dir():
                # Get all selected frames for this video
                frame_files = list(video_dir.glob("*.jpg"))
                
                if frame_files and args.verbose:
                    print(f"Cloning static mask for {len(frame_files)} frames in {video_name}")
                
                # Clone static mask for each selected frame
                for frame_path in frame_files:
                    frame_name = frame_path.stem
                    static_frame_mask = static_video_dir / f"{frame_name}_static.png"
                    shutil.copy2(static_mask_path, static_frame_mask)

    # Vertical flip selected images if requested
    if args.vflip:
        selected_root = output_dir / "selected"
        if args.verbose:
            print("\nApplying vertical flip to selected images...")
        
        for video_name, _ in metadata.items():
            video_dir = selected_root / video_name
            if video_dir.is_dir():
                flip_images_vertical(str(video_dir))

    # Object detection and masking using Detector
    if not args.skip_objects_detection:
        masker = ObjectMasker(model_name='yolov8x.pt', verbose=args.verbose, classes_of_interest=args.detect_class)
        selected_root = output_dir / "selected"
        detection_root = output_dir / "detection"
        
        # Process each video folder
        for video_name, _ in metadata.items():
            input_dir = selected_root / video_name
            if not input_dir.is_dir():
                continue
            out_dir = detection_root / video_name
            out_dir.mkdir(exist_ok=True, parents=True)
            masker.detect(str(input_dir), str(out_dir))
            if args.verbose:
                print(f"Detection and mask results saved to {out_dir}")

    # Vegetation mask generation
    if not args.skip_vegetation:
        veggie_masker = VeggieMasker(area_threshold=args.vegetation_area, verbose=args.verbose)
        selected_root = output_dir / "selected"
        vegetation_root = output_dir / "vegetation"
        
        # Process each video folder
        for video_name, _ in metadata.items():
            input_dir = selected_root / video_name
            if not input_dir.is_dir():
                continue
            out_dir = vegetation_root / video_name
            veggie_masker.process(str(input_dir), str(out_dir), downsample_factor=None)
            if args.verbose:
                print(f"Vegetation masks saved to {out_dir}")

    # Dataset export
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    selected_root = output_dir / "selected"
    detection_root = output_dir / "detection"
    vegetation_root = output_dir / "vegetation"
    static_root = output_dir / "static"

    # Copy selected images
    copy_dir(selected_root, images_dir, files=args.merge)

    # Copy object detection masks  
    if not args.skip_objects_detection and detection_root.exists():
        copy_dir(detection_root, masks_dir, files=args.merge)

    # Copy vegetation masks
    if not args.skip_vegetation and vegetation_root.exists():
        copy_dir(vegetation_root, masks_dir, files=args.merge)

    # Copy static masks
    if static_masks_found and static_root.exists():
        copy_dir(static_root, masks_dir, files=args.merge)

    if args.verbose:
        print(f"Exported images to {images_dir}")
        print(f"Exported masks to {masks_dir}")

    # Combine different mask types for each frame
    if args.verbose:
        print("\nCombining masks for each frame...")
    combine_masks(str(masks_dir))

    # Cleanup temporary files
    if not args.keep_temp_files:
        temps = ["extracted", "selected", "detection", "vegetation", "static"]
        for temp_dir in [output_dir / temp_dir for temp_dir in temps]:
            try:
                shutil.rmtree(temp_dir)
                if args.verbose:
                    print(f"Removed temporary directory: {temp_dir}")
            except FileNotFoundError:
                if args.verbose:
                    print(f"Temporary directory not found (skipped): {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove {temp_dir}: {e}")

def main(args: Args):
    """Main wrapper that handles directory changes and calls the pipeline"""
    # Store current directory
    original_dir = os.getcwd()
    
    try:
        # Determine video directory from the first video path
        if args.video_names:
            video_dir = os.path.dirname(os.path.abspath(args.video_names[0]))
            if args.verbose:
                print(f"Changing to video directory: {video_dir}")
            os.chdir(video_dir)
        
        # Run the pipeline
        pipeline(args)
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)
        if args.verbose:
            print(f"Returned to original directory: {original_dir}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args) 