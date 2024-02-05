"""
Jack Bosco
3/6/2023
"""
from model import ReverseClassifier
from getdata import TorchInvDataset
from torch import nn, Tensor
import torch
import sys
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
"""
First download files for use in custom ImageDataset example
"""

# Load in MNIST dataset from PyTorch
data = np.array([0.0, 1.0])
test_dataset = TorchInvDataset(data)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

#get output file number and train
if len(sys.argv) < 2:
	filename = 'titanic.pt'
else:
	filename = sys.argv[1]
try:
	stateDict = torch.load('outputs'+os.sep+filename)
except:
	raise Exception("No file found: " + filename)

classifier = ReverseClassifier(feature_dim=2048)

def test(test_loader, stateDict, classifier=None):
	classifier.load_state_dict(torch.load('outputs'+os.sep+stateDict))
	classifier.eval()
	result = []

	with torch.no_grad():
		for data, _ in test_loader:
			# print(data)
			# data = data.flatten(start_dim=1)
			out = classifier(data)
			out = list(out)
			
			result.append(1 - out.index(max(out)))
	return result
			

final = test(test_loader=test_dataset, stateDict=filename, classifier=classifier)
print(final)
