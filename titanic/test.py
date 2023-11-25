"""
Jack Bosco
3/6/2023
"""
from model import BaseClassifier
from getdata import TorchOutDataset
from torch import nn, Tensor
import torch
import sys
from torch.utils.data import DataLoader
import os
import pandas as pd
"""
First download files for use in custom ImageDataset example
"""

# Load in MNIST dataset from PyTorch
test_dataset = TorchOutDataset("test_compressed.csv")
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

classifier = BaseClassifier(feature_dim=1024)

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
id = pd.read_csv("kaggle/input/test.csv").PassengerId
df = pd.DataFrame(data={"PassengerId":id, "Survived":final})
df.set_index("PassengerId").to_csv("result.csv")

#utils.printTable(matrix, leftHeaders=False)
#utils.confPlot(matrix=matrix, title='Epochs=40, Hidden_Layers=256')