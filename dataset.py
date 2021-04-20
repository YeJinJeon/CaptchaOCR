import os

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

def remove_file_extension(filename):
    return filename.split('.')[0]

class CapchaDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.image_filenames = os.listdir(base_dir)
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_filepath = os.path.join(self.base_dir, image_filename)
        image = Image.open(image_filepath).convert('RGB')
        image = self.transform(image)
        label = remove_file_extension(image_filename)
        return (image, label)
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        return transform_ops(image)