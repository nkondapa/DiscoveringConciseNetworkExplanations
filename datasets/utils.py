import sys
sys.path.append('../')
import json
import pickle as pkl
import numpy as np
from PIL import Image
import torchvision
import torch
from torchvision.datasets import ImageFolder
import numpy as np
from typing import List, Iterable


# Split data into train, val, and test sets
def split_data(num_samples: int, split_fractions: Iterable[float] = (0.7, 0.15, 0.15),
               seed: int = 0) -> List[np.ndarray]:

    rng = np.random.default_rng(seed)
    inds = rng.permutation(num_samples)

    num_samples_per_split = [int(num_samples * f) for f in split_fractions]
    num_samples_per_split[-1] += num_samples - sum(num_samples_per_split)

    split_inds = []
    start = 0
    for num_samples in num_samples_per_split:
        split_inds.append(inds[start:start + num_samples])
        start += num_samples

    return split_inds


def convert_fraction_to_num_samples(fractions, num_samples):
    num_samples_per_split = [int(num_samples * f) for f in fractions]
    num_samples_per_split[-1] += num_samples - sum(num_samples_per_split)
    return num_samples_per_split


def split_dataset(dataset, split_fractions, seed=42):
    num_samples_per_split = convert_fraction_to_num_samples(split_fractions, len(dataset))
    dataset_splits = torch.utils.data.random_split(dataset, num_samples_per_split,
                                                   generator=torch.Generator().manual_seed(seed))

    return dataset, dataset_splits


def generate_metadata_file(dataset_name, dataset_path):
    print('Generating metadata file')
    if dataset_name == 'CUB':
        root_path = dataset_path

        out = ImageFolder(root_path + 'images/')

        indices = np.arange(len(out.samples))
        subset = np.random.choice(indices, 3000)
        imgs = []
        samples, train_samples, test_samples, train_targets, test_targets = [], [], [], [], []
        resize = torchvision.transforms.Resize((224, 224))

        train_test_split = None
        with open(f'{root_path}/train_test_split.txt', 'r') as f:
            train_test_split = f.readlines()
        train_test_split = [int(line.split(' ')[-1].strip()) for line in train_test_split]

        for si, sample in enumerate(out.samples):
            if si in subset:
                img = Image.open(sample[0]).convert('RGB')
                img = resize(torchvision.transforms.functional.to_tensor(img))
                imgs.append(img)
            s = (sample[0].replace(root_path, ''), sample[1])
            if train_test_split[si] == 0:
                test_samples.append(s)
                test_targets.append(out.targets[si])
            else:
                train_samples.append(s)
                train_targets.append(out.targets[si])

            samples.append(s)

        imgs = torch.stack(imgs)
        mean = imgs.mean(dim=(0, 2, 3)).tolist()
        std = imgs.std(dim=(0, 2, 3)).tolist()
        print(mean, std)

        metadata = {'samples': samples, 'targets': out.targets, 'classes': out.classes,
                    'train_samples': train_samples, 'test_samples': test_samples,
                    'train_targets': train_targets, 'test_targets': test_targets,
                    'class_names': list(map(lambda x: x.split('.')[-1], out.classes)),
                    'class_to_idx': out.class_to_idx, 'mean': mean, 'std': std}

        with open(f'{root_path}/metadata.json', 'w') as f:
            json.dump(metadata, f)