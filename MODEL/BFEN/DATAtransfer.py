import torchvision
import torch
from torchvision import transforms
from .option import Option

crop_size = Option.CROP_SIZE
resize_size=Option.RESIZE_SIZE







train_transforms = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
test_transforms = torchvision.transforms.Compose([
    transforms.Resize(crop_size),
    transforms.CenterCrop(crop_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])

class Normalizetarget2(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample=(sample-5.576186435)/(147.1927382-5.576186435)

        sample =float(sample)
        return torch.tensor(sample)

target_transforms = transforms.Compose([
        Normalizetarget2()
    ])




load_transforms=torchvision.transforms.Compose([
# Generate_patches(),
torchvision.transforms.Resize(resize_size),
    ])
train_transforms=[load_transforms,train_transforms]
test_transforms=[load_transforms,test_transforms]
# target_transforms=None

# patches = generate_patches(img, input_size=self.height)

