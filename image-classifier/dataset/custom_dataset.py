#custom_dataset.py

import os
import shutil
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def setup_directories(meta_file, img_src, raw_out):
    df = pd.read_csv(meta_file)
    types = df['label'].unique()
    for t in types:
        os.makedirs(os.path.join(raw_out, t), exist_ok=True)
    
    for _, item in df.iterrows():
        name = f"{item['id']}.png"
        lbl = item['label']
        src_path = os.path.join(img_src, name)
        dst_path = os.path.join(raw_out, lbl, name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    print(f"Data organized in {raw_out}")

def process_data_split(input_dir, output_dir, split_val=0.8):
    for mode in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, mode), exist_ok=True)
    
    groups = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for g in groups:
        os.makedirs(os.path.join(output_dir, 'train', g), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', g), exist_ok=True)
        
        objs = os.listdir(os.path.join(input_dir, g))
        random.shuffle(objs)
        limit = int(len(objs) * split_val)
        
        for i, obj in enumerate(objs):
            loc = 'train' if i < limit else 'test'
            shutil.copy2(os.path.join(input_dir, g, obj), os.path.join(output_dir, loc, g, obj))
    print(f"Split completed in {output_dir}")

class CIFAR10Dataset(Dataset):
    def __init__(self, folder, transform_node=None):
        self.folder = folder
        self.transform_node = transform_node
        self.classes = sorted(os.listdir(folder))
        self.mapping = {n: i for i, n in enumerate(self.classes)}
        self.archive = []
        for c in self.classes:
            p = os.path.join(folder, c)
            for f in os.listdir(p):
                self.archive.append((os.path.join(p, f), self.mapping[c]))

    def __len__(self):
        return len(self.archive)

    def __getitem__(self, idx):
        path, target = self.archive[idx]
        data = Image.open(path).convert('RGB')
        if self.transform_node:
            data = self.transform_node(data)
        return data, target

if __name__ == "__main__":
    # Internal execution logic
    base_csv = '/content/cifar_dataset/trainLabels.csv'
    base_src = '/content/cifar_dataset/train'
    raw_path = '/content/raw_dataset'
    final_path = '/content/final_dataset'

    if os.path.exists(base_csv):
        setup_directories(base_csv, base_src, raw_path)
        process_data_split(raw_path, final_path)
    else:
        print("Error: Source CSV not found.")
