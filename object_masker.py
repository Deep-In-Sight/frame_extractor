import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from util import downsample, upsample


class ImageFolderDataset(Dataset):
    def __init__(self, folder):
        self.image_paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]


class ObjectMasker:
    def __init__(self, model_name='yolov8x.pt', classes_of_interest=None, verbose=False, debug=False):
        if classes_of_interest is None:
            classes_of_interest = ['person']
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Ensure model uses global cache directory, not current working directory
        if model_name.endswith('.pt') and not os.path.isabs(model_name):
            # Use a fixed global cache directory
            cache_dir = Path.home() / '.ultralytics'
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = cache_dir / model_name
            
            if model_path.exists():
                # Use cached model
                model_name = str(model_path)
            else:
                # Download to cache directory
                if verbose:
                    print(f"Downloading {model_name} to cache: {cache_dir}")
                # Temporarily change to cache dir to download there
                original_cwd = os.getcwd()
                try:
                    os.chdir(str(cache_dir))
                    temp_model = YOLO(model_name, verbose=verbose)
                    # Now the model should be in cache_dir
                    if (cache_dir / model_name).exists():
                        model_name = str(cache_dir / model_name)
                finally:
                    os.chdir(original_cwd)
        
        self.model = YOLO(model_name, verbose=verbose).to(self.device)
        self.verbose = verbose
        self.COCO_CLASSES = self.model.names
        self.target_class_ids = [i for i, name in self.COCO_CLASSES.items() if name in classes_of_interest]
        self.debug = debug

        if self.verbose:
            print(f"ObjectMasker initialized with model {model_name}\n"
                  f"All classes: {self.COCO_CLASSES}\n"
                  f"Classes of interest: {classes_of_interest}\n")

    def detect(self, input_path, output_path, batch_size=4, downsample_factor=None):
        """
        Detect objects in images with optional downsampling for faster processing.
        
        Args:
            input_path (str): Path to input directory with images
            output_path (str): Path to output directory for masks
            batch_size (int): Batch size for processing
            downsample_factor (int, optional): Factor to downsample images by (2, 4, or 8)
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Handle downsampling if requested
        if downsample_factor is not None:
            # Create cache directory for downsampled images
            input_dir = Path(input_path)
            cache_dir = input_dir.parent / f"{input_dir.name}_{downsample_factor}"
            if not cache_dir.exists():
                if self.verbose:
                    print(f"Downsampling images by factor {downsample_factor}")
                downsample(str(input_dir), str(cache_dir), downsample_factor)
            input_path = str(cache_dir)
            
            # Create directory for downsampled results
            output_dir = Path(output_path)
            temp_output_dir = output_dir.parent / f"{output_dir.name}_{downsample_factor}"
            temp_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            temp_output_dir = Path(output_path)
        
        # Process the images (either original or downsampled)
        dataset = ImageFolderDataset(input_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Open a sample image to get size
        sample_image = Image.open(dataset[0]).convert('RGB')
        image_size = sample_image.size

        for batch_paths in tqdm(loader, desc=f"Detecting in {os.path.basename(input_path)}"):
            batch_paths = list(batch_paths)
            batch_results = self.model(batch_paths, verbose=self.verbose)

            for path, result in zip(batch_paths, batch_results):
                img_name = Path(path).stem

                # Prepare empty mask
                mask = Image.new('L', image_size, 255)
                mask_draw = ImageDraw.Draw(mask)

                if self.debug:
                    img_pil = Image.open(path).convert('RGB')
                    draw = ImageDraw.Draw(img_pil)

                for box in result.boxes:
                    cls_id = int(box.cls)
                    if cls_id in self.target_class_ids:
                        xyxy = box.xyxy[0].cpu().numpy()
                        label = self.COCO_CLASSES[cls_id]
                        if self.debug:
                            draw.rectangle(list(xyxy), outline='red', width=5)
                            draw.text((xyxy[0], xyxy[1]), label, fill='red')
                        mask_draw.rectangle(list(xyxy), fill=0)

                # Save outputs
                mask.save(os.path.join(str(temp_output_dir), f"{img_name}_objects.png"))
                if self.debug:
                    img_pil.save(os.path.join(str(temp_output_dir), f"{img_name}_bb.jpg"))

        # Upsample results if we downsampled
        if downsample_factor is not None:
            if self.verbose:
                print(f"Upsampling detection masks by factor {downsample_factor}")
            upsample(str(temp_output_dir), output_path, downsample_factor)

        if self.verbose:
            print("Detection finished.")
