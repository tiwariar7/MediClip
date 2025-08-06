import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import albumentations as A
from tqdm import tqdm
import argparse

class ImagePreprocessor:
    def __init__(self, input_dir="DataSet", output_dir="ProcessedDataSet", 
                 target_size=(224, 224), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Initialize the image preprocessor
        
        Args:
            input_dir: Directory containing Medical and Non-Medical folders
            output_dir: Directory to save processed images
            target_size: Target size for resizing (width, height)
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # ImageNet normalization stats
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Create output directory structure
        self._create_output_dirs()
        
        # Initialize augmentations
        self._setup_augmentations()
        
    def _create_output_dirs(self):
        """Create the output directory structure"""
        splits = ['train', 'val', 'test']
        classes = ['Medical', 'Non-Medical']
        
        for split in splits:
            for class_name in classes:
                split_dir = self.output_dir / split / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
    def _setup_augmentations(self):
        """Setup augmentation pipeline"""
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        ])
        
    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image):
        """Normalize image using ImageNet stats"""
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
            
        return image
    
    def augment_image(self, image):
        """Apply augmentations to image"""
        augmented = self.augmentation(image=image)
        return augmented['image']
    
    def process_single_image(self, image_path, apply_augmentation=True):
        """Process a single image"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = self.resize_image(image)
            
            # Apply augmentation if requested
            if apply_augmentation:
                image = self.augment_image(image)
            
            # Normalize
            image = self.normalize_image(image)
            
            return image
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def get_image_files(self, class_dir):
        """Get all image files from a class directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))
            
        return image_files
    
    def split_dataset(self, image_files):
        """Split dataset into train, validation, and test sets"""
        random.shuffle(image_files)
        
        total_files = len(image_files)
        train_end = int(total_files * self.train_ratio)
        val_end = train_end + int(total_files * self.val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        return train_files, val_files, test_files
    
    def save_processed_image(self, image, output_path):
        """Save processed image"""
        # Convert back to uint8 for saving
        image_uint8 = ((image * self.imagenet_std + self.imagenet_mean) * 255).astype(np.uint8)
        image_uint8 = np.clip(image_uint8, 0, 255)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)
    
    def process_class(self, class_name):
        """Process all images in a class"""
        class_dir = self.input_dir / class_name
        if not class_dir.exists():
            print(f"Class directory {class_dir} does not exist!")
            return
            
        print(f"\nProcessing {class_name} images...")
        image_files = self.get_image_files(class_dir)
        print(f"Found {len(image_files)} images in {class_name}")
        
        if len(image_files) == 0:
            print(f"No images found in {class_name}")
            return
            
        # Split dataset
        train_files, val_files, test_files = self.split_dataset(image_files)
        
        print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Process each split
        splits = [
            ('train', train_files),
            ('val', val_files),
            ('test', test_files)
        ]
        
        for split_name, files in splits:
            print(f"\nProcessing {split_name} split...")
            output_dir = self.output_dir / split_name / class_name
            
            for file_path in tqdm(files, desc=f"{split_name} - {class_name}"):
                # Process image
                processed_image = self.process_single_image(file_path, apply_augmentation=(split_name == 'train'))
                
                if processed_image is not None:
                    # Save processed image
                    output_path = output_dir / file_path.name
                    self.save_processed_image(processed_image, output_path)
    
    def process_dataset(self):
        """Process the entire dataset"""
        print("Starting dataset preprocessing...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target size: {self.target_size}")
        print(f"Split ratios: Train={self.train_ratio}, Val={self.val_ratio}, Test={self.test_ratio}")
        
        # Process each class
        classes = ['Medical', 'Non-Medical']
        for class_name in classes:
            self.process_class(class_name)
            
        print("\nPreprocessing completed!")
        self.print_statistics()
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        splits = ['train', 'val', 'test']
        classes = ['Medical', 'Non-Medical']
        
        for split in splits:
            print(f"\n{split.upper()} SET:")
            total_split = 0
            for class_name in classes:
                split_dir = self.output_dir / split / class_name
                if split_dir.exists():
                    count = len(list(split_dir.glob("*.jpg"))) + len(list(split_dir.glob("*.png")))
                    print(f"  {class_name}: {count} images")
                    total_split += count
            print(f"  Total: {total_split} images")
        
        print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description='Preprocess medical image dataset')
    parser.add_argument('--input_dir', default='DataSet', help='Input directory containing Medical and Non-Medical folders')
    parser.add_argument('--output_dir', default='ProcessedDataSet', help='Output directory for processed images')
    parser.add_argument('--target_size', nargs=2, type=int, default=[224, 224], help='Target image size (width height)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create preprocessor and process dataset
    preprocessor = ImagePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    preprocessor.process_dataset()

if __name__ == "__main__":
    main() 