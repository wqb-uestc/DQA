from MODEL.MODELLOAD.modelmanager import MODELManager
from .loadmodel import LOADModel
from .DATAtransfer import train_transforms, target_transforms, test_transforms
from .option import Option
import sys
import os
os.getcwd()
sys.path.append(os.getcwd()+'/DATA/'+Option.DATASET+'/')
from loaddata import LOADData





class Manager0(MODELManager):
    def __init__(self, options=None,resume=None):
        super(Manager0, self).__init__(options,resume)
    def modelload(self):
        model = LOADModel(self._options, self._path)
        return model
    def dataload(self):
        data = LOADData(self._options, self._path,train_transforms, target_transforms, test_transforms)
        return data


