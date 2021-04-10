##################################################
# Imports
##################################################

from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
import torch
import os
import math
from sklearn.model_selection import train_test_split
import subprocess
from PIL import Image


##################################################
# Utils
##################################################

class AugDataset(Dataset):
    def __init__(self, ds, transform):
        super(AugDataset, self).__init__()
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds.__getitem__(idx)
        x = self.transform(x)
        return x, y
        
def split_train_val(ds, val_ratio, seed=None, stratify=True):
    """
    Split the datasets ds into train and validation datasets given a validation ratio.
    
    Args:
        ds: PyTorch dataset.
        val_ratio: scalar in [0, 1].
        seed: integer for making the split deterministic.
        stratify: bool for making the split balanced among classes.
        
    Output:
        ds_train: PyTorch dataset.
        ds_validation: PyTorch dataset.
    """
    num_validation = int(math.floor(len(ds) * val_ratio))
    idxs_train, idxs_validation = train_test_split(range(len(ds)), test_size=num_validation, random_state=seed, 
                                                   stratify=[y for _, y in ds] if stratify else None,
                                                   shuffle=True)
    return Subset(ds, idxs_train), Subset(ds, idxs_validation)

def subset_by_classes(ds, classes):
    """
    Select a subset of a PyTorch dataset, given the class labels of the samples.
    
    Args:
        ds: PyTorch dataset.
        classes: list of unique labels.
        
    Output:
        ds_sub: PyTorch dataset.
    """
    idxs = [idx for idx, (_, y) in enumerate(ds) if y in classes]
    return Subset(ds, idxs)

def ds_augment(ds, transform):
    """
    Augment the input sample of a dataset ds given a transform function.
    
    Args:
        ds: PyTorch dataset.
        transform: callable function that perturbs a single image of shape [c, h, w].
        
    Output:
        ds_aug: PyTorch dataset.
    """
    return AugDataset(ds, transform)


##################################################
# Tiny Imagenet Dataset
##################################################

class TinyImagenetDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=False):
        super(TinyImagenetDataset, self).__init__()
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self._download()
        self.labels_list = self._retrieve_labels_list()
        self.image_paths, self.labels = self._get_data()

    def _download(self):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        if not os.path.exists(f'{self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip'):
            subprocess.run(f'wget -r -nc -P {self.data_dir} {url}'.split())
            subprocess.run(f'unzip -qq -n {self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip -d {self.data_dir}'.split())

    def _retrieve_labels_list(self):
        labels_list = []
        with open(f'{self.data_dir}/tiny-imagenet-200/wnids.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    labels_list += [line]
        return labels_list

    def _get_data(self):
        image_paths, labels = [], []

        # If train
        if self.train:
            for cl_folder in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train')):
                label = self.labels_list.index(cl_folder)
                for image_name in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images')):
                    image_path = f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images/{image_name}'
                    image_paths += [image_path]
                    labels += [label]

        # If validation
        else:
            with open(f'{self.data_dir}/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    image_name, label_str = line.split('\t')[:2]
                    image_path = f'{self.data_dir}/tiny-imagenet-200/val/images/{image_name}'
                    label = self.labels_list.index(label_str)
                    image_paths += [image_path]
                    labels += [label]
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


##################################################
# Datasets
##################################################

"""
    Each dataset will return samples of the forms (x, y) where:
    - x is the image tensor of shape [channels, height, width].
    - y is the label in forms of string "{source_id}_{absolute_class}_{relative_class}".

        More specifically:
            - source_id: can be "k" for known samples, or "u" for unknown samples.
            - absolute_class: it is the actual class of the dataset.
            - relative_class: it is the index of the class of the dataset relative to the absolute class.
    
            Example:
                dataset: MNIST
                known_classes: [0, 3, 4, 6, 8, 9] 
                    -> absolute_classes: [0, 3, 4, 6, 8, 9] 
                    -> relative_classes: [0, 1, 2, 3, 4, 5]
                unknown_classes: [1, 2, 5, 7]
                    -> absolute_classes: [1, 2, 5, 7] 
                    -> relative_classes: [0, 1, 2, 3]

                    -> a sample from class 9 of the known dataset will be:
                        x: image,
                        y: "k_9_5"

                    -> a sample from class 5 of the unknown dataset will be:
                        x: image,
                        y: "u_5_2"
"""

def get_datasets(args):
    target_transforms = {
        'known': lambda y: f'k_{y}_{args.known_classes.index(y) if y in args.known_classes else -1}',
        'unknown': lambda y: f'u_{y}_{args.unknown_classes.index(y) if y in args.unknown_classes else -1}',
    }

    if args.dataset == 'mnist':

        # Dataset path
        ds_path = os.path.join(args.data_base_path, 'mnist')
        transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1),
        ])
        transform_aug = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1),
        ])

        # Train and validation splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_train, ds_known_validation = split_train_val(
            subset_by_classes(
                MNIST(ds_path, train=True, download=True, transform=None,
                      target_transform=target_transforms['known']),
                classes=known_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )
        ds_known_train_aug = ds_augment(ds_known_train, transform_aug)
        ds_known_train = ds_augment(ds_known_train, transform)
        ds_known_validation = ds_augment(ds_known_validation, transform)
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_train, ds_unknown_validation = split_train_val(
            subset_by_classes(
                MNIST(ds_path, train=True, download=True, transform=transform,
                      target_transform=target_transforms['unknown']),
                classes=unknown_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )

        # Test splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_test = subset_by_classes(
            MNIST(ds_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['known']),
            classes=known_classes,
        )
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_test = subset_by_classes(
            MNIST(ds_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['unknown']),
            classes=unknown_classes,
        )

        # Info 
        height = 32
        width = 32
        channels = 3

    elif args.dataset == 'svhn':

        # Dataset path
        ds_path = os.path.join(args.data_base_path, 'svhn')
        transform = transforms.ToTensor()
        transform_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])

        # Train and validation splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_train, ds_known_validation = split_train_val(
            subset_by_classes(
                SVHN(ds_path, split='train', download=True, transform=None,
                      target_transform=target_transforms['known']),
                classes=known_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )
        ds_known_train_aug = ds_augment(ds_known_train, transform_aug)
        ds_known_train = ds_augment(ds_known_train, transform)
        ds_known_validation = ds_augment(ds_known_validation, transform)
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_train, ds_unknown_validation = split_train_val(
            subset_by_classes(
                SVHN(ds_path, split='train', download=True, transform=transform,
                      target_transform=target_transforms['unknown']),
                classes=unknown_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )

        # Test splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_test = subset_by_classes(
            SVHN(ds_path, split='test', download=True, transform=transform,
                  target_transform=target_transforms['known']),
            classes=known_classes,
        )
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_test = subset_by_classes(
            SVHN(ds_path, split='test', download=True, transform=transform,
                  target_transform=target_transforms['unknown']),
            classes=unknown_classes,
        )

        # Info 
        height = 32
        width = 32
        channels = 3

    elif args.dataset == 'cifar10':

        # Dataset path
        ds_path = os.path.join(args.data_base_path, 'cifar10')
        transform = transforms.ToTensor()
        transform_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # Train and validation splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_train, ds_known_validation = split_train_val(
            subset_by_classes(
                CIFAR10(ds_path, train=True, download=True, transform=None,
                      target_transform=target_transforms['known']),
                classes=known_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )
        ds_known_train_aug = ds_augment(ds_known_train, transform_aug)
        ds_known_train = ds_augment(ds_known_train, transform)
        ds_known_validation = ds_augment(ds_known_validation, transform)
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_train, ds_unknown_validation = split_train_val(
            subset_by_classes(
                CIFAR10(ds_path, train=True, download=True, transform=transform,
                      target_transform=target_transforms['unknown']),
                classes=unknown_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )

        # Test splits
        transform = transforms.ToTensor()
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_test = subset_by_classes(
            CIFAR10(ds_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['known']),
            classes=known_classes,
        )
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_test = subset_by_classes(
            CIFAR10(ds_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['unknown']),
            classes=unknown_classes,
        )

        # Info 
        height = 32
        width = 32
        channels = 3

    elif args.dataset == 'cifar+10':

        # Dataset path
        ds_known_path = os.path.join(args.data_base_path, 'cifar10')
        ds_unknown_path = os.path.join(args.data_base_path, 'cifar100')
        transform = transforms.ToTensor()
        #transform_aug = transforms.ToTensor()
        transform_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # Train and validation splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_train, ds_known_validation = split_train_val(
            subset_by_classes(
                CIFAR10(ds_known_path, train=True, download=True, transform=None,
                      target_transform=target_transforms['known']),
                classes=known_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )
        ds_known_train_aug = ds_augment(ds_known_train, transform_aug)
        ds_known_train = ds_augment(ds_known_train, transform)
        ds_known_validation = ds_augment(ds_known_validation, transform)
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_train, ds_unknown_validation = split_train_val(
            subset_by_classes(
                CIFAR100(ds_unknown_path, train=True, download=True, transform=transform,
                      target_transform=target_transforms['unknown']),
                classes=unknown_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )

        # Test splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_test = subset_by_classes(
            CIFAR10(ds_known_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['known']),
            classes=known_classes,
        )
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_test = subset_by_classes(
            CIFAR100(ds_unknown_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['unknown']),
            classes=unknown_classes,
        )

        # Info 
        height = 32
        width = 32
        channels = 3

    elif args.dataset == 'cifar+50':

        # Dataset path
        ds_known_path = os.path.join(args.data_base_path, 'cifar10')
        ds_unknown_path = os.path.join(args.data_base_path, 'cifar100')
        transform = transforms.ToTensor()
        transform_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # Train and validation splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_train, ds_known_validation = split_train_val(
            subset_by_classes(
                CIFAR10(ds_known_path, train=True, download=True, transform=None,
                      target_transform=target_transforms['known']),
                classes=known_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )
        ds_known_train_aug = ds_augment(ds_known_train, transform_aug)
        ds_known_train = ds_augment(ds_known_train, transform)
        ds_known_validation = ds_augment(ds_known_validation, transform)
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_train, ds_unknown_validation = split_train_val(
            subset_by_classes(
                CIFAR100(ds_unknown_path, train=True, download=True, transform=transform,
                      target_transform=target_transforms['unknown']),
                classes=unknown_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )

        # Test splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_test = subset_by_classes(
            CIFAR10(ds_known_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['known']),
            classes=known_classes,
        )
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_test = subset_by_classes(
            CIFAR100(ds_unknown_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['unknown']),
            classes=unknown_classes,
        )

        # Info 
        height = 32
        width = 32
        channels = 3

    elif args.dataset == 'tiny_imagenet':

        # Dataset path
        ds_path = os.path.join(args.data_base_path, 'tiny_imagenet')
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
        ])
        transform_aug = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
        ])

        # Train and validation splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_train, ds_known_validation = split_train_val(
            subset_by_classes(
                TinyImagenetDataset(ds_path, train=True, download=True, transform=None,
                      target_transform=target_transforms['known']),
                classes=known_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )
        ds_known_train_aug = ds_augment(ds_known_train, transform_aug)
        ds_known_train = ds_augment(ds_known_train, transform)
        ds_known_validation = ds_augment(ds_known_validation, transform)
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_train, ds_unknown_validation = split_train_val(
            subset_by_classes(
                TinyImagenetDataset(ds_path, train=True, download=True, transform=transform,
                      target_transform=target_transforms['unknown']),
                classes=unknown_classes,
            ),
            val_ratio = args.val_ratio,
            seed=args.seed,
            stratify=True,
        )

        # Test splits
        known_classes = [target_transforms['known'](cl) for cl in args.known_classes]
        ds_known_test = subset_by_classes(
            TinyImagenetDataset(ds_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['known']),
            classes=known_classes,
        )
        unknown_classes = [target_transforms['unknown'](cl) for cl in args.unknown_classes]
        ds_unknown_test = subset_by_classes(
            TinyImagenetDataset(ds_path, train=False, download=True, transform=transform,
                  target_transform=target_transforms['unknown']),
            classes=unknown_classes,
        )

        # Info 
        height = 32
        width = 32
        channels = 3

    # Datasets
    dss = {
        'known': {
            'train': ds_known_train,
            'train_aug': ds_known_train_aug,
            'validation': ds_known_validation,
            'test': ds_known_test,
        },
        'unknown': {
            'train': ds_unknown_train,
            'validation': ds_unknown_validation,
            'test': ds_unknown_test,
        },
        'train': ConcatDataset([ds_known_train, ds_unknown_train]),
        'validation': ConcatDataset([ds_known_validation, ds_unknown_validation]),
        'test': ConcatDataset([ds_known_test, ds_unknown_test]),
    }
    dss_info = {
        'height': height,
        'width': width,
        'channels': channels,
    }
    return dss, dss_info


##################################################
# Data Loaders
##################################################

def get_dataloaders(args):

    # Datasets
    dss, dss_info = get_datasets(args)

    # Dataloaders
    dls_args = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dls = {
        'known': {
            'train': DataLoader(dss['known']['train'], shuffle=False, **dls_args),
            'train_aug': DataLoader(dss['known']['train_aug'], shuffle=True, **dls_args),
            'validation': DataLoader(dss['known']['validation'], shuffle=False, **dls_args),
            'test': DataLoader(dss['known']['test'], shuffle=False, **dls_args),
        },
        'unknown': {
            'train': DataLoader(dss['unknown']['train'], shuffle=False, **dls_args),
            'validation': DataLoader(dss['unknown']['validation'], shuffle=False, **dls_args),
            'test': DataLoader(dss['unknown']['test'], shuffle=False, **dls_args),
        },
        'train': DataLoader(dss['train'], shuffle=False, **dls_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dls_args),
        'test': DataLoader(dss['test'], shuffle=False, **dls_args),
    }
    return dls, dss_info
