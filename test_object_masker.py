from object_masker import ObjectMasker
import os
from pathlib import Path

def main():
    masker = ObjectMasker(model_name='yolov8x.pt', verbose=True)
    input_root = Path('output/selected')
    output_root = Path('output/masks')
    if not input_root.exists():
        print(f'No input directory {input_root}')
        return
    # For each subfolder (e.g., vid0, vid1, ...)
    for subfolder in input_root.iterdir():
        if subfolder.is_dir():
            image_paths = [str(p) for p in subfolder.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if not image_paths:
                print(f'No images found in {subfolder}')
                continue
            out_dir = output_root / subfolder.name
            os.makedirs(out_dir, exist_ok=True)
            masker.detect(str(subfolder), str(out_dir))
            print(f'Detection and mask results saved to {out_dir}')

if __name__ == '__main__':
    main() 