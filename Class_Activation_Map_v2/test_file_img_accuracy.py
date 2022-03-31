import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from copy_network_CNN import *

#________________________gpu사용여부
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
#___________________________________

batch_size = 10

#___________________________dataload
#Tensor로 이미지 변환
trans = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor()
    ])
#data file 가져오기
test_data = torchvision.datasets.ImageFolder(root='image/test_data',transform=trans)
#test data를 가져오기
test_set = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)
#___________________________________

#network 소환
test_net = CNN().to(device)
#가중치 소환
test_net.load_state_dict(torch.load('weight/class5/model_c5_(5e-05)_(14)_(94.26).pth'))


with torch.no_grad():
    for num, data in enumerate(test_set):
        imgs, label = data
        imgs = imgs.to(device)
        label = label.to(device)
        prediction,f = test_net(imgs)

        #label과 prediction 값이 일치하면 1 아니면 0
        correct_prediction = torch.argmax(prediction,1) == label
        print(torch.argmax(prediction,1))
        accuracy = correct_prediction.float().mean()
        
        print('Accuracy: ', accuracy.item())