import cv2
import numpy as np
import time

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
from pandas import DataFrame #엑셀저장

from copy_network_CNN import *


#________________________gpu사용여부
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

#_________________________함수설정
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

trans = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor()
    ])

#__________________________test 이미지 불러오기
cap_image = cv2.imread('image/raw_img/v3_34.jpg') # my image path


slideSize = 112
w_width, w_height = 224,224 # window size
print('가로 : ', cap_image.shape[1])
print('세로 : ', cap_image.shape[0])

CAM_row = (cap_image.shape[1]//slideSize)*slideSize
CAM = np.zeros(((cap_image.shape[0]//slideSize)*slideSize,CAM_row))

cam_net = CNN().to(device)
cam_net.load_state_dict(torch.load('weight/class4/model_c4_(5e-05)_(30)_(98.4).pth'))

#cam_net.load_state_dict(torch.load('weight/class5/model_c5_(5e-05)_(21)_(98.93).pth'))

start = time.time()

#___________________________slide window를 통한 영역별 classification 실행
for x in range(0, (cap_image.shape[1] - w_width), slideSize):
    
    for y in range(0, (cap_image.shape[0] - w_height), slideSize):
        window = cap_image[y:(y+w_height), x:(x+w_width), :]
        print(window.shape)
        
        ten_win = trans(window) # 224*224 이미지를 tensor로 변환

        #print('_______학습_________')
        with torch.no_grad():
            imgs = ten_win
            imgs = imgs.to(device)
            imgs = imgs.unsqueeze(dim=0)
            prediction, f = cam_net(imgs)
            
            answer = torch.argmax(prediction,1)
            print('x:',x,'~',x+w_width,', y:',y,'~',y+w_height, '영역은 ', answer, '번 영역입니다')
            if answer == 1:
                CAM[y:(y+w_height), x:(x+w_width)] += 1
                
        

    params = list(cam_net.parameters())

            
print("cost time : ", time.time()-start)

#데이터 시각화
img = trans(cap_image) 
imshow(img) 
plt.imshow(CAM, alpha=0.3,cmap='jet')
plt.show()



