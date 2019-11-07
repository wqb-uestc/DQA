class LOADModelO(object):
    def __init__(self, options,path):
        self.solver=solver
        self.exp_lr_scheduler=scheduler
        self.criterion = criterion
        self._options=options
        self._path=path
        self.load()
        print(self.net)

    def load(self):
        self._net()
        self._criterion()
        self._optimize()
    def _criterion(self):
        pass
    def _net(self):
        pass

    def _optimize(self):
        pass

class solver(object):
    pass
class criterion(object):
    pass
class scheduler(object):
    pass


