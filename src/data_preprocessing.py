import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os

class RoadSawDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or self._get_default_transforms()
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self):
        """Get all image paths from the data directory."""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        return [
            f for f in self.data_dir.rglob('*')
            if f.suffix.lower() in valid_extensions
        ]
    
    @staticmethod
    def _get_default_transforms():
        """Default preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Standard size for many models
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path.name

def prepare_dataset(raw_data_path, processed_data_path):
    """Prepare and save processed dataset."""
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Create dataset instance
    dataset = RoadSawDataset(raw_data_path)
    
    # Process and save each image
    for idx in range(len(dataset)):
        image, filename = dataset[idx]
        # Save processed tensor
        torch.save(image, Path(processed_data_path) / f"{filename}.pt")

if __name__ == "__main__":
    prepare_dataset("data/raw", "data/processed")
