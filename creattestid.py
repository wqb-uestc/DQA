import random
index = list(range(1, 207))
with open('test_id.txt','w') as f :
    for i in range(10):
        random.shuffle(index)
        test_index = index[164:len(index)]
        f.write(' '.join(list(map(lambda x: '{:}'.format(x), test_index))) + '\n')
