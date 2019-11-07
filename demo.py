import os
from DATA.utils import pil_loader
from MODEL.BFEN.model import bfen as BFEN
import torch
from torchvision import transforms



test_transforms = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def main():
    path = 'derain-data'
    imgpath = [os.path.join(path, i) for i in os.listdir(path)]

    resume = '%s_%s.pth.tar' % ('BFEN', 'pretrain')
    bfen = BFEN().cuda()
    checkpoint = torch.load(resume,map_location='cuda:0')
    bfen.load_state_dict(checkpoint['state'])


    bfen.train(False)
    scoreall=[]
    with torch.no_grad():
        for path in imgpath:
            x = pil_loader(path)
            x = test_transforms(x)
            x = x.unsqueeze(0).cuda()
            score = bfen(x)
            score=score.item()
            scoreall.append(score)
            # print('{:}:{:.4f}'.format(path.split('\\')[1],score))

        print('{:.4f}'.format(float(sum(scoreall)) / len(scoreall)))
        with open('evaluation_result.txt', 'w') as f:
            f.write('\n'.join(list(map(lambda x,y: '{:}:{:.4f}'.format(os.path.split(x)[1],y),imgpath ,scoreall))))

if __name__ == '__main__':
    main()
