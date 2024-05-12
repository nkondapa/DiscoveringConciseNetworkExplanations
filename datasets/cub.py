
from PIL import Image
import os
import os.path
import torch.utils.data
import numpy as np
import json
import datasets.utils as utils

def default_image_loader(path):

    image = Image.open(path).convert('RGB')

    return image


class CUBDataset(torch.utils.data.Dataset):

    def __init__(self, root_path, transforms=None, target_transforms=None, loader=default_image_loader, split=None,
                 class_subset=None, dataset_subset=None, return_path=False, **kwargs):

        self.loader = loader
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.root_path = root_path
        self.return_path = return_path

        if not os.path.isfile(self.root_path + 'metadata.json'):
            utils.generate_metadata_file('CUB', self.root_path)

        with open(self.root_path + 'metadata.json', 'r') as f:
            metadata = json.load(f)
            self.metadata = metadata
            self.samples = metadata['samples']
            self.train_samples = metadata['train_samples']
            self.test_samples = metadata['test_samples']
            self.train_targets = metadata['train_targets']
            self.test_targets = metadata['test_targets']
            self.targets = metadata['targets']

            self.mean = metadata['mean']
            self.std = metadata['std']

            self.classes = metadata['classes']  # if classes are not reader friendly we also have class_names
            self.class_names = metadata['class_names']  # class_names maps classes to a more reader friendly name
            self.class_to_idx = metadata['class_to_idx']
            self.idx_to_class = metadata['class_to_idx']
            self.num_classes = len(self.class_to_idx)

        # different ways to split the dataset (first two are user defined), second two are standard splits
        if dataset_subset is not None:
            self.random_subset(dataset_subset, keep_all_in_classes=class_subset)
        elif class_subset is not None:
            for cs in class_subset:
                print(self.class_names[cs])
            self.samples = [sample for sample in self.samples if sample[1] in class_subset]
            self.targets = [target for target in self.targets if target in class_subset]
        elif split == 'train':
            self.samples = self.train_samples
            self.targets = self.train_targets
        elif split == 'test':
            self.samples = self.test_samples
            self.targets = self.test_targets

            # self.class_names = [class_name for class_name in self.class_names if class_name in class_subset]
            # self.class_to_idx = {class_: idx for idx, class_ in enumerate(self.classes)}

        # ADD ROOT PATH TO SAMPLES
        for i in range(len(self.samples)):
            self.samples[i][0] = os.path.join(self.root_path, self.samples[i][0])

        self.num_classes = len(self.class_to_idx)
        if self.transforms is not None and hasattr(self.transforms.transforms[0], 'size'):
            self.default_image_resize = self.transforms.transforms[0].size

    def get_class_subset_indices(self, class_subset):
        if type(class_subset) is not list:
            class_subset = [class_subset]

        sample_indices = [idx for idx, sample in enumerate(self.samples) if sample[1] in class_subset]
        return sample_indices

    def random_subset(self, subset_size, keep_all_in_classes=None):
        if subset_size is None:
            return

        if len(self) > subset_size:
            tmp_indices = np.arange(len(self))
            np.random.shuffle(tmp_indices)

            if keep_all_in_classes is not None:
                class_sample_indices = self.get_class_subset_indices(keep_all_in_classes)
                tmp_indices = np.concatenate([class_sample_indices, tmp_indices[~np.isin(tmp_indices, class_sample_indices)]], axis=0)
                tmp_indices = tmp_indices[:subset_size]
                np.random.shuffle(tmp_indices)
            else:
                tmp_indices = tmp_indices[:subset_size]

            sub_samples = []
            sub_targets = []
            for ind in tmp_indices:
                sub_samples.append(self.samples[ind])
                sub_targets.append(self.targets[ind])

            self.samples = sub_samples
            self.targets = sub_targets

    def __getitem__(self, index):
        image_path, target = self.samples[index][0], self.samples[index][1]
        image = self.loader(image_path)

        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        if self.return_path:
            return image, target, image_path

        return image, target

    def __len__(self):
        return len(self.samples)