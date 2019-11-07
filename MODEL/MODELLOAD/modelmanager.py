import os
import torch
from scipy import stats
import time
# from torch.utils.tensorboard.writer import SummaryWriter


class MODELManager(object):
    def __init__(self, options=None,resume=None):
        if options is None:
            pass
        else:
            print('Prepare the network and data.')
            self._options = options
            self._path = options.path
            # Network.
            model = self.modelload()
            if resume is None:
                self._net = model.net
            else:
                self._net = model.net
                checkpoint = torch.load(resume)
                self._net.load_state_dict(checkpoint['state'])

            self._best_srcc = AverageMeter('best_srcc', ':6.4f')
            self._best_srcc.val = -1
            if os.path.isdir(self._options.MODEL):
                try:
                    resume = '%s_best%d.pth.tar' % (self._options.MODEL, self.shuju)
                    resume = os.path.join(self._options.MODEL, resume)
                    checkpoint = torch.load(resume)
                    self._best_srcc.val = checkpoint['srocc']
                    print('===> Load last checkpoint data')
                except FileNotFoundError:
                    print('Can\'t found autoencoder.t7')
            else:
                self._best_srcc.val = 0
                print('===> Start from scratch')


            self._criterion=model.criterion
            self._solver=model.solver
            self._scheduler=model.exp_lr_scheduler
            data = self.dataload()
            self._train_loader = data.train_loader
            self._test_loader = data.test_loader
            # parameter
            self._epoch=None
            self._loss=[]
            self._st=None

            self._train_srcc=0
            self._test_srcc=0
            self._test_plcc=0
            self._test_krocc=0
            # self.writer = SummaryWriter(comment=self._options.MODEL+str(self.shuju))
            self.AverageMeter=AverageMeter
            self.ProgressMeter=ProgressMeter
    def dataload(self):
        pass
    def modelload(self):
        pass

    def val(self):
        self._net.train(False)
        self._test_srcc = AverageMeter('test_srcc', ':6.2f')
        self._test_plcc = AverageMeter('test_plcc', ':6.2f')
        self._test_krocc = AverageMeter('test_krocc', ':6.2f')
        progress = ProgressMeter(len(self._test_loader),
                                      self._test_srcc, self._test_plcc, self._test_krocc,
                                      prefix=self._epoch)


        self._pscores = []
        self._tscores = []
        for x, y, _ in self._test_loader:
            x = x.cuda()
            y = y.cuda()
            score = self.score(x)

            self._pscores = self._pscores + score.cpu().tolist()
            self._tscores = self._tscores + y.cpu().tolist()

        self._test_srcc.update(stats.spearmanr(self._pscores, self._tscores)[0])
        self._test_plcc.update(stats.pearsonr(self._pscores, self._tscores)[0])
        self._test_krocc.update(stats.kendalltau(self._pscores, self._tscores)[0])
        self._net.train(True)
        progress.print(len(self._test_loader))

    def train(self):

        self._best_plcc = AverageMeter('best_prcc', ':6.4f')
        self._best_krocc = AverageMeter('best_krcc', ':6.4f')
        self._epoch_time = AverageMeter('Time', ':6.3f')

        self._train_srcc = AverageMeter('train_srcc', ':6.2f')
        self._epochend = time.time()
        for self._epoch in range(self._options.EPOCHS):

            self._batch_time = AverageMeter('Time', ':6.3f')
            self._data_time = AverageMeter('Data', ':6.3f')
            self._loss = AverageMeter('Loss', ':6.4f')
            progress = ProgressMeter(len(self._train_loader),
                                          self._epoch_time, self._batch_time, self._data_time,
                                          self._loss, self._train_srcc,self._best_srcc,self._best_plcc,self._best_krocc,
                                          prefix=self._epoch)
            end = time.time()
            self._pscores = []
            self._tscores = []

            self.schedulstep()

            for x, y, _ in self._train_loader:
                self._data_time.update(time.time() - end)
                x = x.clone().detach().cuda()
                y = y.clone().detach().cuda().float()
                loss,score=self.backloss(x,y)


                self._loss.update(loss.item(), x.size(0))
                self._pscores = self._pscores + score.cpu().tolist()
                self._tscores = self._tscores + y.cpu().tolist()
                self._batch_time.update(time.time() - end)
                end = time.time()

            self._train_srcc.update(stats.spearmanr(self._pscores, self._tscores)[0])
            if self._epoch % 1 == 0:
                self.val()
                self._epoch_time.update(time.time() - self._epochend)
                self._epochend = time.time()
                progress.print(len(self._train_loader))
                # progress.plotloss(self.writer)
            if self._test_srcc.val > self._best_srcc.val:
                self._best_srcc.update(self._test_srcc.val)
                self._best_plcc.update(self._test_plcc.val)
                self._best_krocc.update(self._test_krocc.val)
                self._best_epoch = self._epoch
                self._modelsave()


    def _modelsave(self):
        state = {
            'state': self._net.state_dict(),
            'srocc': self._test_srcc.val
        }
        filename = '%s_best%d.pth.tar' % (self._options.MODEL, self.shuju)
        filename = os.path.join(self._options.MODEL, filename)
        if not os.path.isdir(self._options.MODEL):
            os.mkdir(self._options.MODEL)
        torch.save(state, filename)
























class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    def plotstr(self):
        fmtstr ='{name}'
        return {fmtstr.format(**self.__dict__):self.val}


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = ["Epoch: [{}]".format(self.prefix) + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def plotloss(self,writer):
        for meter in self.meters:
            writer.add_scalars(meter.name, meter.plotstr(), self.prefix)