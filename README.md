B-FEN source code release.

Authors     : Qingbo Wu, Lei Wang, King N. Ngan, Hongliang Li, Fanman Meng, 
              Linfeng Xu
Version     : Beta1.0

The authors are with the School of Electric Engineering, University of Electronic Science and Technology of China, Chengdu 611731, P. R. China.

Existing methods typically evaluate de-raining performance on a few synthesized rain images, whose rain-free images are available. Then, two classic full-reference objective metrics including PSNR and SSIM are employed for quantitative quality assessment.In comparison with the diverse authentic rain images, these limited synthetic data are far from sufficient to verify 
the de-raining capability in reality.

This research aims to develop a reliable blind quality assessment model for the de-rained image, which facilitates the quantitative evaluation on the authentic rain images. 

===========================================================================

-------------------------COPYRIGHT NOTICE----------------------------------

Copyright (c) 2019 University of Electronic Science and Technology of China
All rights reserved.

Rights to all images are retained by the photographers. For researchers and educators who wish to use the images for non-commercial research and/or educational purposes, we can provide access under the following terms:

1. Researcher shall use the Database only for non-commercial research and educational purposes.
2. University of Electronic Science and Technology of China makes no representations or warranties regarding the Database, including but not limited to warranties of non-infringement or fitness for a particular purpose.
3. Researcher accepts full responsibility for his or her use of the Database and shall defend and indemnify University of Electronic Science and Technology of China, including their employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the Database.
4. Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions.
5. University of Electronic Science and Technology of China reserves the right to terminate Researcher's access to the Database at any time.
6. If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.

---------------------------Instructions------------------------------------

This is a PyTorch implementation of the proposed B-FEN model. If this code is helpful for your research, please cite the following papers in your bibliography, i.e.,

1. Q. Wu, L. Wang, K. N. Ngan, H. Li and F. Meng, "Beyond Synthetic Data: A Blind Deraining Quality Assessment Metric Towards Authentic Rain Image," IEEE International Conference on Image Processing, 2019, pp. 2364-2368.

2. Q. Wu, L. Wang, K. N. Ngan, H. Li, F. Meng and L. Xu, "Subjective and Objective De-raining Quality Assessment Towards Authentic Rain Image," arXiv preprint arXiv:1909.11983, 2019.

## Requirements

Please install the necessary packages

- Python 3.6+
- pytorch 1.2.0
- numpy 
- scipy 
- torchvision 
- PIL
- future

   
------------------------------Contact Info------------------------------------

If you have any suggestions or problems in the usage of this database, please 
feel free to contact qbwu@uestc.edu.cn


## Requirements

Please install requirements 

- Python 3.6+
- pytorch 1.2.0
- numpy 
- scipy 
- torchvision 
- PIL
- future

## Usage

- If you are interested in evaluating the performance of your de-raining algorithm:

(1) Please download our pre-trained DQA model (https://drive.google.com/open?id=190vuiTEsF5KmwKV8iXmGLFE_3Jjdgp_O) and put it into the root directory of '\DQA-master'.

(2) put all your de-rained images into the 'derain-data' folder and directly run the script of 'demo.py'. The qualities of all de-rained images would be saved  in 'evaluation-result.txt' and the mean quality of all images would be shown in your runner window. The reported scores range from 0 to 1, and a higher score means better quality.

- If you are interested in developing the DQA model:

(1) Please download the IVIPC-DQA database (https://drive.google.com/open?id=1Pe17CX0WN3kK3GH18femUyrW2sBYDMVl) and put it into the directory of '\DQA-master\dataset` .

(2) Run 'creattestid.py' to randomly separate the IVIPC-DQA database into the 
non-overlapped training and testing sets. The ID of all test images would be 
saved in the file of 'test_id.txt'.

It is noted that the default values of 'test_id.txt' are used for replicating the results reported in our paper. The users could run 'creattestid.py' to override 'test_id.txt' and generate your custom training and testing sets.

(3) Run 'dqa.py' to conduct 10 times of training and testing trials on 'test_id.txt'. 

### Opinion-unaware Metrics 

Run ou-metric.m in matlab to get a result of NIQE,QAC,LPSI,ILNIQE algorithm in 
'OU metric` folder .
