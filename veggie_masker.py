import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from util import downsample, upsample

class VeggieMasker:
    def __init__(self, area_threshold=800, verbose=False):
        self.area_threshold = area_threshold
        self.verbose = verbose

    def process_image(self, image_path, output_path):
        """Process a single image and save its vegetation mask"""
        # Load the image
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define green color range in HSV
        lower_green = np.array([35, 20, 40])
        upper_green = np.array([85, 255, 255])

        # Create initial green mask
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Invert the mask so vegetation (black) is removed in Metashape
        mask_inverted = cv2.bitwise_not(green_mask)

        # Morphological closing to fill small holes
        kernel = np.ones((10, 10), np.uint8)
        mask_closed = cv2.morphologyEx(mask_inverted, cv2.MORPH_CLOSE, kernel)

        # Remove small white noise blobs by connected component filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
        refined_mask = np.zeros_like(mask_closed)
        for i in range(1, num_labels):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= self.area_threshold:
                refined_mask[labels == i] = 255

        # Final morphological cleanup
        final_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

        # Save result as PNG
        cv2.imwrite(output_path, final_mask)

    def process(self, input_dir, output_dir, downsample_factor=None):
        """
        Process all images in a directory in parallel and save vegetation masks.
        
        Args:
            input_dir (str): Source directory containing original images
            output_dir (str): Destination directory for vegetation masks
            downsample_factor (int, optional): Factor to downsample images by (2, 4, or 8)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Handle downsampling if requested
        if downsample_factor is not None:
            # Create cache directory for downsampled images
            cache_dir = input_dir.parent / f"{input_dir.name}_{downsample_factor}"
            if not cache_dir.exists():
                if self.verbose:
                    print(f"Downsampling images by factor {downsample_factor}")
                downsample(str(input_dir), str(cache_dir), downsample_factor)
            input_dir = cache_dir
            
            # Create directory for downsampled results
            temp_output_dir = output_dir.parent / f"{output_dir.name}_{downsample_factor}"
            temp_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            temp_output_dir = output_dir
            
        temp_output_dir.mkdir(exist_ok=True, parents=True)

        # Get all jpg images
        image_files = list(input_dir.glob("*.jpg"))
        if not image_files:
            if self.verbose:
                print(f"No jpg images found in {input_dir}")
            return

        def process_single_image(img_path):
            output_path = temp_output_dir / f"{img_path.stem}_vegetation.png"
            try:
                self.process_image(str(img_path), str(output_path))
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        # Process images in parallel with tqdm's thread_map
        thread_map(process_single_image, image_files,
                  desc=f"Processing vegetation in {input_dir.name}",
                  max_workers=None)
            
        # Upsample results if we downsampled
        if downsample_factor is not None:
            if self.verbose:
                print(f"Upsampling vegetation masks by factor {downsample_factor}")
            upsample(str(temp_output_dir), str(output_dir), downsample_factor)

# Keep the function for backward compatibility
def mask_green_vegetation(image_path, output_path, area_threshold=800):
    processor = VeggieMasker(area_threshold=area_threshold)
    processor.process_image(image_path, output_path)
