import os
import json

from torch.utils.data import Dataset
from PIL import Image 

class ClassificationDataset(Dataset):
    def __init__(self, cfg, data_dir, label_path, transform):
        super(ClassificationDataset, self).__init__()
        self.dataset_name = cfg.dataset.name
        self.data_dir = data_dir
        self.label_path = label_path
        self.transform = transform

        self.data_filename = []
        for path in os.listdir(data_dir):
            if not path.endswith('.png') and not path.endswith('.jpg'):
                raise ValueError(f"Unsupported file format in {data_dir}: {path}")
            # self.data.append(os.path.join(data_dir, path))
            self.data_filename.append(path)
        
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            self.labels = label_data

    def __len__(self):
        return len(self.data_filename)

    def __getitem__(self, idx):
        # return self.data[idx], self.labels[idx]
        data_path = os.path.join(self.data_dir, self.data_filename[idx])
        label = self.labels[self.data_filename[idx]]
        image = Image.open(data_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
        


