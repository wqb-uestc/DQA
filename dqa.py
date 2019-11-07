
from MODEL.BFEN.manager import Manager as BFEN
def main():

#load net
    bfen = BFEN()
#begin to train 10 times and write result
    bfen.trainall()


#begin to test 10 times and write result
    bfen.test()



if __name__ == '__main__':
    main()

