import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt
from Model import CNN
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


trn_dataset=datasets.MNIST('../mnist_data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()#image to tensor
,transforms.Normalize((0.1307,),(0.3081,)) ])) #image, label

val_dataset=datasets.MNIST('../mnist_data', download=False, train=False, transform=transforms.Compose([transforms.ToTensor()#image to tensor
,transforms.Normalize((0.1307,),(0.3081,)) ])) #image, label


test_dataset=datasets.MNIST('../mnist_data', train=False, transform=transforms.Compose([transforms.ToTensor()#image to tensor
,transforms.Normalize((0.1307,),(0.3081,)) ])) #image, label

#for batch processing
batch_size=64
trn_loader=torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

classes=('0','1','2','3','4','5','6','7','8','9')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


use_cuda=torch.cuda.is_available()

criterion=nn.CrossEntropyLoss()
#backpropagation method
cnn=CNN()
learning_rate=1e-3
optimizer=optim.Adam(cnn.parameters(), lr=learning_rate)
#hyper-parameters
num_epochs=2
num_batches=len(trn_loader)

trn_loss_list=[]
val_loss_list=[]

for epoch in range(num_epochs):
    trn_loss=0.0
    for i, data in enumerate(trn_loader):
        x, label= data
        if use_cuda:
            x=x.cuda()
            label=label.cuda()

        #grad init
        optimizer.zero_grad()
        #foward propagation
        model_output=cnn(x)
        #calculate loss
        loss=criterion(model_output,label)
        #back propagation
        loss.backward()
        #weight update
        optimizer.step()

        #trn_loss summary
        trn_loss+=loss.item()
        #del (memory issue)
        del loss
        del model_output

        #print training process
        if (i+1)%100==0: #every 100 mini-batches
            with torch.no_grad(): #very important
                val_loss=0.0
                for j, val in enumerate(val_loader):
                    val_x, val_label=val
                    if use_cuda:
                        val_x=val_x.cuda()
                        val_label=val_label.cuda()
                    val_output=cnn(val_x)
                    v_loss=criterion(val_output, val_label)
                    val_loss=v_loss

            print("epoch: {}/{}  | step: {}/{}  | trn loss: {:.4f}  |val loss: {:.4f}".format(epoch+1, num_epochs, i+1, num_batches, trn_loss/100, val_loss/len(val_loader)))

            trn_loss_list.append(trn_loss/100)
            val_loss_list.append(val_loss/len(val_loader))
            trn_loss= 0.0

#print images
detaiter=iter(test_loader)
images, labels=detaiter.next()

imshow(torchvision.utils.make_grid(images))
print('Groundtruth: '.join('%5s'% classes[labels[j]] for j in range(4)))
plt.show()
outputs=cnn(images)
_, predicted=torch.max(outputs,1)
print('predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        images, labels=data
        outputs=cnn(images)
        _, predicted=torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print('Accuracy of the network : %d %%' % (100*correct/total))
