import torch
from DATA.utils import *
__all__ = ['LOADDataO',]
class LOADDataO(object):
    def __init__(self, options,path):
        self._options = options
        self._path = path
        self._datapath = None
        self._train_data = None
        self._test_data = None
        self.train_loader = None
        self.test_loader = None
        self._folder1 = None
        self._folder2 = None
        self.datapath()
        self.dataload()
    def trans(self):
        pass
    def dataload(self):
        self.folder()
        self._train_data = self._folder1(
            root=self._datapath, loader=default_loader, index=self._options.TRAIN_INDEX,
            transform=self._train_transforms, target_transform=self._target_transforms, extensions=IMG_EXTENSIONS)
        self._test_data = self._folder2(
            root=self._datapath, loader=default_loader, index=self._options.TEST_INDEX,
            transform=self._test_transforms, target_transform=self._target_transforms, extensions=IMG_EXTENSIONS)
        self.train_loader = torch.utils.data.DataLoader(
            self._train_data, batch_size=self._options.BATCH_SIZE,
            shuffle=True, num_workers=0, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            self._test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)
    def datapath(self):
        pass
    def folder(self):
        pass

