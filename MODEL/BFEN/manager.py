from .manager0 import Manager0
import os
from .option import Option
import torch


class Manager(Manager0):
    def __init__(self, options=None,resume=None):
        super(Manager, self).__init__(options,resume)

    # fun of score
    def score(self,x):
        score = self._net(x)
        score = score.view(x.size(0))
        return score
    # fun of lossback
    def backloss(self,x,y):

        self._solver.optimizer.zero_grad()
        score = self._net(x)
        loss = self._criterion.score(score.view(1, -1), y)
        loss.backward()
        self._solver.optimizer.step()
        return loss,score
    def schedulstep(self):
        self._scheduler.optimizer.step()
    def test(self):
        # run 10 times
        for self.shuju in range(10):

            #option
            self._options = Option(self.shuju)
            resume='%s_best%d.pth.tar' % (self._options.MODEL,self.shuju)
            resume = os.path.join(self._options.MODEL, resume)
            #initial
            self.__init__(options=self._options,resume=resume)

            #begin test
            self.val()

            #write result
            with open(self._options.MODEL+'/pscore.txt', 'a') as f:
                f.write(str(self.shuju) + ' ')
                [f.write(str(x)+' ') for x in self._pscores]
                f.write('\n')
            with open(self._options.MODEL+'/tscore.txt', 'a') as f:
                f.write(str(self.shuju) + ' ')
                [f.write(str(x)+' ') for x in self._tscores]
                f.write('\n')
            with open(self._options.MODEL+'/testresult.txt', 'a') as f:
                f.write(str(self.shuju)+' '+str(self._test_srcc.val)+' '+str(self._test_plcc.val)+' '+str(self._test_krocc.val))
                f.write('\n')
            with open(self._options.MODEL+'/target.txt', 'a') as f:
                f.write(str(self.shuju) + ' ')
                [f.write(str(target+1) + ' ') for path, target, imgindex, score  in self._test_loader.dataset.samples]
                f.write('\n')
            with open(self._options.MODEL+'/index.txt', 'a') as f:
                f.write(str(self.shuju) + ' ')
                [f.write(str(imgindex) + ' ') for path, target, imgindex, score  in self._test_loader.dataset.samples]
                f.write('\n')


    def trainall(self):
        #run 10 times
        for self.shuju in [0,2,3,4,5,7,9]:
            # option
            self._options = Option(self.shuju)
            # initial
            self.__init__(self._options)
            #begin train
            self.train()
            # write result
            with open(self._options.MODEL + '/result.txt', 'a') as f:
                f.write(str(self.shuju) + ' ' + str(self._best_srcc.val) + ' ' + str(self._best_plcc.val) + ' ' + str(
                    self._best_krocc.val))
                f.write('\n')

