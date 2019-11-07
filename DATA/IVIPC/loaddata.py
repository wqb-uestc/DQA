from DATA.DATALOAD.DATAload import *
from DATA.IVIPC.DATAFolder import IVIPCFolder


class LOADData(LOADDataO):

    def __init__(self, options,path,train_transforms,target_transforms,test_transforms):
        self._train_transforms = train_transforms
        self._target_transforms = target_transforms
        self._test_transforms = test_transforms
        super(LOADData, self).__init__(options,path)





    def datapath(self):
        self._datapath = self._path[self._options.DATASET]

    def folder(self):
        self._folder1 = IVIPCFolder
        self._folder2 = IVIPCFolder
