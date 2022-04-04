import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from network_CNN import *

import visdom 
#실행법 : python -m visdom.server
"""
vls = visdom.Visdom()
vls.close(env="main")

Y_data = 0 
plt = vls.line(Y=Y_data)

X_data = 0
plt = vls.line(Y=Y_data, X=X_data, env='main')
"""
if __name__ == '__main__':    
    #________________________gpu사용여부
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    #___________________________________

    batch_size = 30
    learning_rate = 0.00005
    epochs = 70

    #___________________________dataload
    #Tensor로 이미지 변환
    trans = transforms.Compose([transforms.ToTensor()])
    #data file 가져오기
    train_data = torchvision.datasets.ImageFolder(root='images/train_data',transform=trans)
    #data random으로 batch_size만큼 가져오기
    data_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    #___________________________________

    #__________________________

    net = CNN().to(device)
    #network size test
    test_input = (torch.Tensor(3,3,224,224).to(device))
    test_out = net(test_input)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss().to(device)

    total_batch = len(data_loader)

    for epoch in range(epochs):
        avg_cost = 0
        for num, data in enumerate(data_loader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out, f = net(imgs)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch
        
        print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))

        """
        Y_data = 0 
        plt = vls.line(Y=Y_data)

        X_data = 0
        plt = vls.line(Y=Y_data, X=X_data, env='main')

        Y_append = epoch+1
        X_append = avg_cost
        plt = vls.line(Y=Y_append,X=X_append,win=plt,update='append',env='main')
        """
    print('learning Finished!')

    #가중치 저장
    torch.save(net.state_dict(), "model({})_({}).pth".format(learning_rate, epochs))

