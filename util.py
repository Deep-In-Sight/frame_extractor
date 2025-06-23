import cv2
import os
import numpy as np
from pathlib import Path
from tqdm.contrib.concurrent import thread_map
from functools import partial

def _process_image_downsample(args):
    filename, src_dir, dst_dir, factor = args
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return
        
    # Read image
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return
        
    # Calculate new dimensions by division
    new_width = img.shape[1] // factor
    new_height = img.shape[0] // factor
    
    # Resize image
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save downsampled image
    output_path = os.path.join(dst_dir, filename)
    cv2.imwrite(output_path, resized)

def _process_image_upsample(args):
    filename, src_dir, dst_dir, factor = args
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return
        
    # Read image as grayscale since it's a binary mask
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return
    
    # Calculate new dimensions by multiplication
    new_width = img.shape[1] * factor
    new_height = img.shape[0] * factor
    
    # Resize using nearest neighbor interpolation for binary masks
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Ensure the result is binary (0 or 255)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    # Save upsampled binary mask
    output_path = os.path.join(dst_dir, filename)
    cv2.imwrite(output_path, resized)

def downsample(src_dir, dst_dir, factor, max_workers=None):
    """
    Downsample images in the source directory by dividing dimensions by the factor.
    Uses parallel processing for faster execution.
    
    Args:
        src_dir (str): Source directory containing original images
        dst_dir (str): Destination directory for downsampled images
        factor (int): Downsampling factor (must be 2, 4, or 8)
        max_workers (int, optional): Maximum number of worker threads. Defaults to None (auto).
    """
    assert factor in [2, 4, 8], "Downsampling factor must be 2, 4, or 8"
        
    # Create destination directory if it doesn't exist
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of files
    files = os.listdir(src_dir)
    
    # Create arguments for each file
    args = [(f, src_dir, dst_dir, factor) for f in files]
    
    # Process images in parallel with progress bar
    thread_map(_process_image_downsample, args, 
              desc=f"Downsampling images by 1/{factor}", 
              max_workers=max_workers)

def upsample(src_dir, dst_dir, factor, max_workers=None):
    """
    Upsample binary masks in the source directory by multiplying dimensions by the factor.
    Uses parallel processing for faster execution.
    
    Args:
        src_dir (str): Source directory containing original masks
        dst_dir (str): Destination directory for upsampled masks
        factor (int): Upsampling factor (must be 2, 4, or 8)
        max_workers (int, optional): Maximum number of worker threads. Defaults to None (auto).
    """
    assert factor in [2, 4, 8], "Upsampling factor must be 2, 4, or 8"
        
    # Create destination directory if it doesn't exist
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of files
    files = os.listdir(src_dir)
    
    # Create arguments for each file
    args = [(f, src_dir, dst_dir, factor) for f in files]
    
    # Process images in parallel with progress bar
    thread_map(_process_image_upsample, args, 
              desc=f"Upsampling masks by {factor}x", 
              max_workers=max_workers)

def _process_image_flip(img_path):
    """Flip a single image vertically and overwrite it"""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is not None:
            # 180 degree rotation
            flipped_img = cv2.flip(img, -1)
            # Overwrite the original image
            cv2.imwrite(str(img_path), flipped_img)
    except Exception as e:
        print(f"Error flipping {img_path}: {e}")

def flip_images_vertical(src_dir, max_workers=None):
    """
    Vertically flip all images in the source directory in-place.
    Uses parallel processing for faster execution.
    
    Args:
        src_dir (str): Source directory containing images to flip
        max_workers (int, optional): Maximum number of worker threads. Defaults to None (auto).
    """
    src_path = Path(src_dir)
    
    # Get all jpg images
    image_files = list(src_path.glob("*.jpg"))
    if not image_files:
        return
    
    # Process images in parallel with progress bar
    thread_map(_process_image_flip, image_files, 
              desc=f"Flipping {src_path.name}",
              max_workers=max_workers)

def combine_masks(dir_name, max_workers=None):
    """
    Recursively combine different mask types for the same frame using boolean AND operation.
    Looks for masks in format: {frame_name}_{mask_type}.png throughout the directory tree.
    Combines all mask types for each frame and saves as {frame_name}.png in the same location.
    
    Args:
        dir_name (str): Root directory to search for mask files
        max_workers (int, optional): Maximum number of worker threads. Defaults to None (auto).
    """
    masks_dir = Path(dir_name)
    if not masks_dir.exists():
        return
    
    # Group masks by directory and frame name
    mask_groups = {}
    
    # Recursively find all PNG mask files
    for mask_path in masks_dir.rglob("*.png"):
        mask_name = mask_path.stem
        
        # Skip if it's already a combined mask (no underscore suffix)
        if '_' not in mask_name:
            continue
            
        # Extract frame name (everything before the last underscore)
        frame_name = '_'.join(mask_name.split('_')[:-1])
        
        # Use the parent directory as part of the grouping key
        group_key = (mask_path.parent, frame_name)
        
        if group_key not in mask_groups:
            mask_groups[group_key] = []
        mask_groups[group_key].append(mask_path)
    
    def _combine_frame_masks(group_data):
        """Combine all masks for a single frame in a specific directory"""
        (directory, frame_name), mask_paths = group_data
        
        if not mask_paths:
            return
            
        try:
            # Load the first mask to get dimensions
            first_mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
            if first_mask is None:
                print(f"Warning: Could not read mask {mask_paths[0]}")
                return
                
            # Start with the first mask
            combined_mask = first_mask.copy()
            
            # AND with all other masks
            for mask_path in mask_paths[1:]:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Ensure same dimensions
                    if mask.shape != combined_mask.shape:
                        mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                    # Boolean AND operation
                    combined_mask = cv2.bitwise_and(combined_mask, mask)
                else:
                    print(f"Warning: Could not read mask {mask_path}")
            
            # Save combined mask in the same directory as the source masks
            output_path = directory / f"{frame_name}.png"
            cv2.imwrite(str(output_path), combined_mask)
            
        except Exception as e:
            print(f"Error combining masks for frame {frame_name} in {directory}: {e}")
    
    if mask_groups:
        # Process frame groups in parallel
        group_items = list(mask_groups.items())
        thread_map(_combine_frame_masks, group_items,
                  desc="Combining masks",
                  max_workers=max_workers) 