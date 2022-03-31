import torch
import torch.nn as nn 
import torch.nn.functional as F 

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from copy_network_CNN import *

import visdom 
import time

#실행법 : python -m visdom.server
def tracker(loss_plot, loss_value, num):
    vis.line(X=num,Y=loss_value,win=loss_plot,update='append', env="main")


if __name__ == '__main__':    
    #________________________gpu사용여부
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    #___________________________________

    vis = visdom.Visdom()
    vis.close(env="main")
    accuracy_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='train_accuracy',legend=['train_accuracy'], showlegend=True))
    val_accuracy_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='val_accuracy',legend=['val_accuracy'], showlegend=True))
    loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='train_loss',legend=['loss'], showlegend=True))
    val_loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='val_loss',legend=['loss'], showlegend=True))
    

    batch_size = 45
    test_batch_size = 22
    learning_rate = 0.00005
    epochs = 50

    #___________________________dataload
    #Tensor로 이미지 변환
    trans = transforms.Compose([transforms.ToTensor()])
    #data file 가져오기
    train_data = torchvision.datasets.ImageFolder(root='image/train_data',transform=trans)
    test_data = torchvision.datasets.ImageFolder(root='image/test_data',transform=trans)
    
    #data random으로 batch_size만큼 가져오기
    data_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = DataLoader(dataset = test_data, batch_size=test_batch_size, shuffle=True)
    
    
    #___________________________________
    net = CNN().to(device)
    #network size test
    test_input = (torch.Tensor(3,3,224,224).to(device))
    test_out = net(test_input)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss().to(device)

    total_batch = len(data_loader)
    val_total_batch = len(test_set)

    
    #____________________________training
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
        
        #_____________________test
        with torch.no_grad():
            val_avg_cost = 0
            t = 0
            sum_val_accuracy = 0
            for num, data in enumerate(test_set):
                imgs, label = data
                imgs = imgs.to(device)
                label = label.to(device)
                prediction,f = net(imgs)

                val_loss = loss_func(prediction, label)
                val_avg_cost += val_loss/val_total_batch

                #label과 prediction 값이 일치하면 1 아니면 0
                correct_prediction = torch.argmax(prediction,1) == label

                sum_val_accuracy += correct_prediction.float().mean()
                t += 1

            avg_accuracy = sum_accuracy / n
            val_accuracy = sum_val_accuracy / t
        
        print('[Epoch:{}] cost time : {}, cost : {}, train_Accuracy : {}, val_accuracy : {}'.format(epoch+1, time.time()-start, avg_cost, avg_accuracy.item(), val_accuracy.item()))

        #___________________________시각화
        tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch+1]))
        tracker(val_loss_plt, torch.Tensor([val_avg_cost]), torch.Tensor([epoch+1]))
        tracker(accuracy_plt, torch.Tensor([avg_accuracy.item()]), torch.Tensor([epoch+1]))
        tracker(val_accuracy_plt, torch.Tensor([val_accuracy.item()]), torch.Tensor([epoch+1]))

        if avg_cost <= 0.02:
            torch.save(net.state_dict(), "./weight/class4/model_c4_({})_({})_({}).pth".format(learning_rate, epoch, round(val_accuracy.item()*100,2)))
        elif val_accuracy >= 0.95:
            torch.save(net.state_dict(), "./weight/class4/model_c4_({})_({})_({}).pth".format(learning_rate, epoch, round(val_accuracy.item()*100,2)))
        
        
    print('learning Finished!')

    #가중치 저장
    torch.save(net.state_dict(), "./weight/class4/model_c4_({})_({})_({}).pth".format(learning_rate, epochs, round(val_accuracy.item()*100,2)))


