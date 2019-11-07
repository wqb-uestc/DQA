import torch.utils.data as data

from PIL import Image
import scipy.io as scio
import os
import os.path
# import math
import numpy as np
import random
import re

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir ,extensions,huafen,class_to_idx):
    os.path.split(dir)
    scores=scio.loadmat(os.path.join(os.path.split(dir)[0],'realigned_mos.mat'))['realigned_mos']

    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    index = re.findall(r"\d+", fname)
                    imgindex = int(index[0])
                    if imgindex  in huafen:
                        path = os.path.join(root, fname)

                        # imgindex=int(index[0])
                        tar=class_to_idx[target]
                        item = (path, target,imgindex,scores[imgindex-1,tar])
                        images.append(item)

    return images

class DATAFolder(data.Dataset):
    def __init__(self, root, loader, index, transform=None, target_transform=None,  extensions=None ):

        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions, index,class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.scores = [x[3] for x in samples]
        self.transform = transform
        self.target_transform = target_transform

    def getsample(self, index):
        path, target, imgindex, score = self.samples[index]
        img = self.loader(path)
        score=score.astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            score = self.target_transform(score)

        return img, score

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img, score = self.getsample(index)

        return img,score

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



