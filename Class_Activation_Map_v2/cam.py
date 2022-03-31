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

from network_CNN import *

#________________________gpu사용여부
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
#___________________________________

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


#___________________________dataload
#Tensor로 이미지 변환
trans = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor()
    ])
#data file 가져오기
test_data = torchvision.datasets.ImageFolder(root='images/test_data',transform=trans)
#test data를 가져오기
test_set = DataLoader(dataset = test_data, batch_size=10, shuffle=True)

#___________________________________

cam_net = CNN().to(device)
cam_net.load_state_dict(torch.load('model(7e-05)_(28)_(95.82).pth'))

with torch.no_grad():
    for data in test_set:
        
        imgs, label = data
        print(imgs.shape)
        print(imgs)
        print('____________')
        imgs = imgs.to(device)
        label = label.to(device)
        prediction,f = cam_net(imgs)
        

        #label과 prediction 값이 일치하면 1 아니면 0
        correct_prediction = torch.argmax(prediction,1)
        break
print('____________')
print(correct_prediction)

"""
for data in testloader:    
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs, f = cam_net(images)
    _, predicted = torch.max(outputs, 1)
    break
"""

classes =  ('ground', 'obstacle', 'sky', 'sprinkler', 'tree')
params = list(cam_net.parameters())
print(params[0])
num = 0
for num in range(10):
    print("ANS :",classes[int(correct_prediction[num])]," REAL :",classes[int(label[num])],num)

    #print(outputs[0])
    #overlay = params[-2][int(correct_prediction[num])].matmul(f[num].reshape(512,49)).reshape(7,7).cpu().data.numpy()
    overlay = params[-2][int(correct_prediction[num])].matmul(f[num].reshape(512,49)).reshape(7,7).cpu().data.numpy()
    #print(overlay)
    
    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay)
    #print(overlay)
    
    print(imgs[num].cpu().shape)
    imshow(imgs[num].cpu())
    #skimage.transform.resize(overlay, [224,224])
    plt.imshow(skimage.transform.resize(overlay, [224,224]), alpha=0.3,cmap='jet')
    plt.show()
    #imshow(imgs[num].cpu())


