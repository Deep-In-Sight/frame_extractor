import os
from pathlib import Path
import tyro
from dataclasses import dataclass
from veggie_masker import VeggieMasker

@dataclass
class Args:
    input_dir: str = "output/images"  # Directory containing input images
    output_dir: str = "output/masks"  # Directory to save vegetation masks
    area_threshold: int = 800        # Minimum area for vegetation patches
    verbose: bool = False            # Verbose output

def main(args: Args):
    # Convert to Path objects for easier handling
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get list of all jpg images
    image_files = list(input_dir.glob("*.jpg"))
    if not image_files:
        print(f"No jpg images found in {input_dir}")
        return
        
    if args.verbose:
        print(f"Found {len(image_files)} images to process")
    
    # Process each image
    masker = VeggieMasker(area_threshold=args.area_threshold, verbose=args.verbose)
    masker.process(str(input_dir), str(output_dir))
            
    if args.verbose:
        print(f"\nCompleted vegetation mask generation")
        print(f"Masks saved to: {output_dir}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args) 