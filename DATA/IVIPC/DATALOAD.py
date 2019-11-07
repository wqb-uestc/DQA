from DATA.IVIPC.DATAFolder import DATAFolder
from DATA.DATALOAD.DATALOAD import DATALoaderO


class DATALoader(DATALoaderO):

    def __init__(self, options,path):
        super(DATALoader, self).__init__(options,path)

    def trans(self):
        pass

    def folder(self):
        self._folder = DATAFolder
    def datapath(self):
        pass

