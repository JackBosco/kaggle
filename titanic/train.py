"""
Jack Bosco
3/6/2023
"""
from model import BaseClassifier
from getdata import TorchDataset
from torch import nn, optim
import torch.cuda
from torch.utils.data import DataLoader
import sys
from matplotlib import pyplot as plt
import os
"""
First download files for use in custom ImageDataset example
"""

# Load in MNIST dataset from PyTorch

train_dataset = TorchDataset("compressed.csv")
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Instantiate model, optimizer, and hyperparameter(s)
in_dim, feature_dim, out_dim = 25, 512, 2
loss_fn = nn.CrossEntropyLoss()
epochs=4000
lr =  1e-3

classifier = BaseClassifier(in_dim, feature_dim, out_dim)

dev = 'cuda' if torch.cuda.is_available else 'cpu'
dev = 'cpu'
print('running on ' + dev)
print('training for', epochs, 'epochs')
print('learning rate:', lr)

classifier = classifier.to(dev)

#get output file number and train
if len(sys.argv) < 2:
    ext = ''
else:
    ext = sys.argv[1]
    
def train(epochs, batch_size\
        , train_loader, filenum = '', plot=False, lr = 1e-3\
        , classifier=None, loss_fn=None):
    classifier.train()
    optimizer = optim.SGD(classifier.parameters(), lr=lr)
    loss_lt = []
    for epoch in range(epochs):
        running_loss = 0.0
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1)

            data = data.to(dev)
            target = target.to(dev)

            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Keep track of sum of loss of each minibatch
            running_loss += computed_loss.item()
        loss_lt.append(running_loss/len(train_loader))
        if epoch % 200 == 0:
        	print("Epoch: {} train loss: {:.4f}".format(epoch+1, running_loss/len(train_loader)))
    
    if plot:
        plt.plot([i for i in range(1,epochs+1)], loss_lt)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(
                "MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))

    # Save state to file as checkpoint
    torch.save(classifier.state_dict(), 'outputs{0}titanic{1}.pt'.format(os.sep, filenum))
    plt.savefig('outputs{0}titanic{1}.png'.format(os.sep,filenum))

train(plot=True,epochs=epochs, batch_size=64, train_loader=train_loader, filenum=ext, lr=lr, classifier=classifier, loss_fn=loss_fn)
