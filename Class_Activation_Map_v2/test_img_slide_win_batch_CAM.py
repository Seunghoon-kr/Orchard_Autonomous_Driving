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
import openpyxl

from copy_network_CNN import *


#________________________gpu사용여부
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
#___________________________________
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


trans = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor()
    ])



cap_image = cv2.imread('image/v1_8.jpg') # my image path
print()
slideSize = 112
(w_width, w_height) = (224,224) # window size
#print(cap_image.shape[1])
print(cap_image.shape[0])

CAM_row = (cap_image.shape[1]//slideSize)*slideSize
CAM = np.zeros(((cap_image.shape[0]//slideSize)*slideSize,CAM_row))

batch_size = cap_image.shape[0]//slideSize -1
count = 0


test_set = torch.zeros([batch_size,3,w_width,w_height])
cam_net = CNN().to(device)
cam_net.load_state_dict(torch.load('weight/model_c5_(5e-05)_(63)_(97.47).pth'))

start = time.time()

for y in range(0, (cap_image.shape[1] - 224), slideSize):
    for x in range(0, (cap_image.shape[0] - 224), slideSize):
        window = cap_image[x:(x+w_width), y:(y+w_height), :]
        
        ten_win = trans(window) 
        test_set[count] = ten_win
        count += 1

    with torch.no_grad():
        imgs = test_set
        imgs = imgs.to(device)
        prediction,f = cam_net(imgs)
        
        #label과 prediction 값이 일치하면 1 아니면 0
        correct_prediction = torch.argmax(prediction,1)
        

    params = list(cam_net.parameters())
    #print('_____________________________')
    for num in range(batch_size):
        
        #print(outputs[0])
        overlay2 = params[-4].matmul(f.reshape(512,196))
        overlay1 = params[-4].matmul(overlay2)
        print(overlay1.shape)
        overlay = params[-2][0].matmul(overlay1).reshape(14,14).cpu().data.numpy()
        overlay = skimage.transform.resize(overlay, [224,224])
        CAM[num*slideSize:(num*slideSize)+w_width, y:(y+w_height)] += overlay
        
        #plt.imshow(CAM, alpha=0.3,cmap='jet')
        #plt.show()


    count = 0
    #test_set = torch.zeros([batch_size,3,w_width,w_height])
            
print("cost time : ", time.time()-start)
'''
raw_data = {}
for i in range(CAM_row):
    raw_data['{}'.format(i)] = CAM.T[i]
data = DataFrame(raw_data)
#print(data)
data.to_excel('CAM_data.xlsx')
'''
cv2.imwrite('cam1.jpg', CAM)


img = trans(cap_image) 
imshow(img) #[0:(cap_image.shape[0]//slideSize)*slideSize,0:(cap_image.shape[1]//slideSize)*slideSize]
plt.imshow(CAM, alpha=0.3,cmap='jet')
plt.show()


