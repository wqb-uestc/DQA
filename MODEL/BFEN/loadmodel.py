import torch
import torch.nn as nn
from MODEL.MODELLOAD.MODELLoad import LOADModelO
from torch.optim import lr_scheduler
from .model import bfen

class LOADModel(LOADModelO):
    def __init__(self, options,path):
        super(LOADModel, self).__init__(options,path)
    def _criterion(self):
        self.criterion.score =nn.PairwiseDistance(p=2).cuda()
    def _optimize(self):
        lr_decay = self._options.LR_DECAY
        decay_step = self._options.DECAY_STEP
        weight_decay = self._options.WEIGHT_DECAY


        self.solver.optimizer = torch.optim.SGD(self.net.parameters(),lr=self._options.LEARNING_RATE,momentum=0.9, weight_decay=weight_decay)
        self.exp_lr_scheduler.optimizer = lr_scheduler.StepLR(self.solver.optimizer, step_size=decay_step,
                                                    gamma=lr_decay)
    def _net(self):
        self.net = torch.nn.DataParallel(bfen(pretrained=True), device_ids=[0]).cuda()




