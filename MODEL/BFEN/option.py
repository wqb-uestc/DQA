from MODEL.option import Config

import os
class Option(Config):
    WEIGHT_DECAY=0
    RESIZE_SIZE=320
    CROP_SIZE=320

    LEARNING_RATE =0.01
    BATCH_SIZE=2
    EPOCHS = 35
    LR_DECAY=0.8
    DECAY_STEP=7
    MODEL = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
    DATASET = 'IVIPC'
    TRAIN_INDEX = 0
    TEST_INDEX = 1
    save_freq=100
    resume_epoch=0
    shuju='None'

    def __init__(self,i=None):
        super(Option, self).__init__()

        if i== None:
            pass

        a=[]
        if self.DATASET == 'IVIPC' :
            with open('test_id.txt') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.split()
                    a_float_m = map(int, line)
                    a_float_m = list(a_float_m)
                    a.append(a_float_m)
            test_index = a[i]
            train_index = list(range(1,207))
            condition = lambda t: t not in test_index
            filter_list = list(filter(condition, train_index))
            train_index = filter_list



        else :
            raise Exception('wrong')
        self.TRAIN_INDEX = train_index
        self.TEST_INDEX = test_index

        self.TRAIN_INDEX = train_index
        self.TEST_INDEX = test_index

