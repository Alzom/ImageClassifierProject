

import my_transforms
import numpy as np 
from PIL import Image



# null_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# #load random image data
# data = ImageFolder(root = "C:\\Users\\moroz\\Documents\\Projects\\Pytorch Classifier\data\\plants\\dataset\\images\\lab", transform = data_transform)
# sampler = RandomSampler(data)
# loader = DataLoader(data, sampler=sampler)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, CenterCrop
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import Dataset 
import my_transforms
import numpy as np 
from PIL import Image

# transforms
# data_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(size=(500,600)),
#     my_transforms.RandomRot(),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])

#Data location
train_dir = ".\\data\\plants\\dataset\\images\\labtest"
test_dir = ".\\data\\plants\\dataset\\images\\labval"

lr = 0.01 #learn rate
batch = 4 #batch size
epochs = 1 #number of epochs 
pad = 1 #padding

batchnorm_momentum = 0.9

cuda = torch.cuda.is_available() #checks if cuda is available


if __name__ == '__main__':

    #transforms
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(500),
        my_transforms.RandomRot(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    null_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(500),
        transforms.ToTensor()
    ])
    #training set
    train_set = ImageFolder(root = train_dir , transform = data_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)
    
    

    #validation set
    valid_set = ImageFolder(root = test_dir , transform = null_transform)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch, shuffle=True)
    

    def conv3x3(inputs, outputs, pad, maxpool):
        layer = nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size = 3, stride =1, padding = pad),
            nn.BatchNorm2d(outputs, momentum = batchnorm_momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = maxpool,stride = maxpool))
        return layer

    def dense(inputs, outputs):
        layer = nn.Sequential(
            nn.Linear(inputs, outputs),
            nn.BatchNorm1d(outputs, momentum = batchnorm_momentum),
            nn.ReLU())
        return layer 

    def final(inputs, outputs):
        return nn.Linear(inputs, outputs)

    class PNet(nn.Module):
        def __init__(self):
            super(PNet, self).__init__()
            self.center = nn.BatchNorm2d(num_features=3, momentum=batchnorm_momentum)
            self.conv1 = conv3x3(3, 16, pad, 3)
            self.conv2 = conv3x3(16, 32, pad, 2)
            self.conv3 = conv3x3(32, 64, pad, 2)
            self.conv4 = conv3x3(64, 64, pad, 3)
            self.conv5 = conv3x3(64, 32, pad, 3)
            self.fc1 = dense(512, 256)
            self.fc2 = dense(256, 128)
            self.final = final(128, 4)
        
        def forward(self, x): 
            x = self.center(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.final(x)
            return x

    network = PNet()
    network.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    

    best_val_acc = 0.

    for epoch in range(epochs):
        print('\n\nEpoch {}'.format(epoch))

        ### training loss/accuracy

        network.train()
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            if i % 20 == 0: print('Training batch {} of {}'.format(i, len(train_loader)))
            if cuda:
                images = images.cuda()
                labels = labels.cuda()

            images = Variable(images)
            labels = Variable(labels)
            

            optimizer.zero_grad()
            output = network(images)
            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        accuracy = correct / total
        print('Mean train acc over epoch = {}'.format(accuracy))

        ### validation loss/accuracy

        network.eval()
        print('Validation')
        total = 0
        correct = 0
        for images, labels in valid_loader:
            if cuda:
                images = images.cuda()
                labels = labels.cuda()

            images = Variable(images)
            labels = Variable(labels.long())

            output = network(images)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        accuracy = correct / total
        print('Mean val acc over epoch = {}'.format(accuracy))
        if accuracy > best_val_acc: best_val_acc = accuracy
        print('Best val acc = {}'.format(best_val_acc))











        
