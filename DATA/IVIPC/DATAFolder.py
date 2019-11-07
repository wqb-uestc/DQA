import torch.utils.data as data
import scipy.io as scio
import os
import os.path
import numpy as np
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
    scores=scio.loadmat(os.path.join(dir,'realigned_mos.mat'))['realigned_mos']

    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)[:6]):
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

                        tar=class_to_idx[target]
                        item = (path, tar,imgindex,scores[imgindex-1,tar])
                        images.append(item)

    return images

class IVIPCFolder(data.Dataset):
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
        self.transform = transform[1]
        self.target_transform = target_transform
        self.img=[]

        for index in range(len(self.samples)):
            path, target, imgindex, score = self.samples[index]
            img=loader(path)
            img=transform[0](img)
            self.img.append(img)
    def __getitem__(self, index):
        path, target, imgindex, score = self.samples[index]
        img = self.img[index]
        score = score.astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            score = self.target_transform(score)
        return img, score ,target-1
    def __len__(self):
        return len(self.samples)
