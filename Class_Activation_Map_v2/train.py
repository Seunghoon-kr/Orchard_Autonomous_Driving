import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from network_CNN import *

import visdom 
import time
'''
#실행법 : python -m visdom.server

vis = visdom.Visdom()
#vls.close(env="main")
Y_data = torch.Tensor([0])
#plt = vis.line(Y=Y_data)
X_data = torch.Tensor([0])
plt = vis.line(Y=Y_data, X=X_data, env='main')
'''


if __name__ == '__main__':    
    #________________________gpu사용여부
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    #___________________________________

    batch_size = 48
    learning_rate = 0.00005
    epochs = 100

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
        start = time.time()
        n = 0
        sum_accuracy = 0
        for num, data in enumerate(data_loader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out, f = net(imgs)
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch
            
            correct_prediction = torch.argmax(out,1) == labels
            
            accuracy = correct_prediction.float().mean()
            
            n += 1 
            sum_accuracy += accuracy

        avg_accuracy = sum_accuracy / n
        
        print('[Epoch:{}] cost time : {}, cost : {}, Accuracy : {}'.format(epoch+1, time.time()-start, avg_cost, avg_accuracy.item()))

        '''
        #___________________________시각화
        X_append = epoch+1
        Y_append = torch.cat((avg_cost, avg_accuracy), dim=1)
        plt = vis.line(Y=Y_append,X=X_append,win=plt,update='append',env='main')
        '''               
        
    print('learning Finished!')

    #가중치 저장
    torch.save(net.state_dict(), "model({})_({})_({}).pth".format(learning_rate, epochs, avg_cost))

    