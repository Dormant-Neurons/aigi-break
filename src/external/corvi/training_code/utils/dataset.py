'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from torch.utils.data import Dataset
from PIL import Image
from .processing import make_processing, add_processing_arguments
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PathNameDataset(ImageFolder):
    def __init__(self, **keys):
        super().__init__(**keys)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": sample, "target": target, "path": path}


def get_dataset(opt, dataroot):
    dset_lst = []
    # NOTE: get the classes for the current directory
    if os.path.isdir(os.path.join(dataroot, "0_real")):
        classes = ['.',]
    else:
        classes = os.listdir(dataroot)

    transform = make_processing(opt)
    print(classes)
    for cls in classes:
        root = dataroot + "/" + cls
        if os.path.isdir(root + "/0_real"):
            dset = PathNameDataset(root=root, transform=transform)
            print("#images %6d in %s" % (len(dset), root))
            dset_lst.append(dset)

    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    if torch.all(w==w[0]):
        print(f"RandomSampler: # {ratio}")
        sampler = RandomSampler(dataset, replacement = False)
    else:
        w = w / torch.sum(w)
        print(f"WeightedRandomSampler: # {ratio}, Weightes {w}")
        sample_weights = w[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    return sampler


def add_dataloader_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for dataloader
    parser.add_argument(
        "--dataroot", type=str, help="Path to the dataset to use during training"
    )
    # The path containing the train and the validation data to train on

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_threads", default=8, type=int, help='# threads for loading data')
    parser = add_processing_arguments(parser)
    return parser


def create_dataloader(opt, subdir='.', is_train=True):
    dataroot = os.path.join(opt.dataroot, subdir)
    dataset = get_dataset(opt, dataroot)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=get_bal_sampler(dataset) if is_train else None,
        num_workers=int(opt.num_threads),
    )
    return data_loader




class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Make labels consistent: real=0, fake=1
        self.classes = ['real', 'fake']  
        self.class_to_idx = {'real': 0, 'fake': 1}  # Don't override this later
        print(f"Dataset initialized with class mapping: {self.class_to_idx}")
        self.images = []
        self.labels = []
        self.paths = []
        
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
                
            # Iterate through model subdirectories
            for model_name in os.listdir(class_dir):
                model_dir = os.path.join(class_dir, model_name)
                if os.path.isdir(model_dir):
                    # Look for the 'nature' symlink or other category directories
                    for category in os.listdir(model_dir):
                        category_path = os.path.join(model_dir, category)
                        if os.path.islink(category_path) or os.path.isdir(category_path):
                            # Follow symlink and get actual image files
                            real_path = os.path.realpath(category_path)
                            if os.path.exists(real_path):
                                for img_name in os.listdir(real_path):
                                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        img_path = os.path.join(real_path, img_name)
                                        self.images.append(img_path)
                                        self.labels.append(self.class_to_idx[cls])
                                        self.paths.append(img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, img_path

def create_dataloaders_genimage(opt, batch_size=32, valid_size=0.001, num_workers=64, pin_memory=True):
    """
    Creates training and validation DataLoaders.

    Parameters:
        opt (object): Options or configuration object for making transforms.
        path (str): Path to the dataset.
        batch_size (int): Number of samples per batch.
        valid_size (float): Fraction of the dataset to be used for validation.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to copy tensors into CUDA pinned memory.

    Returns:
        tuple: (train_loader, valid_loader)
    """
    
    # Define transforms for the training and validation datasets
    transform = make_processing(opt)  # Ensure this function is defined elsewhere

    # Load the dataset
    dataset = CustomDataset(root_dir=opt.dataroot, transform=transform)
    
    # Add diagnostic prints
    print(f"Dataset size: {len(dataset)}")
    print(f"Label distribution: {np.bincount([l for l in dataset.labels])}")
    
    # Calculate the split indices
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(valid_size * num_train)
    
    # Randomly shuffle the indices and split
    torch.manual_seed(42)  # Set seed for reproducibility
    indices = torch.randperm(num_train)
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # Create dataset subsets
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    
    # Define a custom collate function to format data into dictionaries
    def custom_collate(batch):
        imgs, lbls, paths = zip(*batch)
        return {'img': torch.stack(imgs), 'target': torch.tensor(lbls), 'path': paths}
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=custom_collate)
    
    return train_loader, valid_loader
