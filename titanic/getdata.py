"""
Jack Bosco
Kaggle Titanic Challange with Convolutional NN
"""

import numpy as np  # for linear algebra
import pandas as pd # for I/O
from torch.utils.data import Dataset
from torch import Tensor


def compress(filepath):
	"""
	Takes a file, applies the data compression for the titanit probem, returns a dataframe
	@param filepath
	@return compress dataframe
	"""
	df = pd.read_csv(filepath)

	df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1) # drop the unimportant columns by label

	df.loc[df["Embarked"] == 'Q', "Embarked"] = 0.0 				 #
	df.loc[df["Embarked"] == 'S', "Embarked"] = 0.5 				 # Change "Embarked" to numeric values
	df.loc[df["Embarked"] == 'C', "Embarked"] = 1.0                  # 

	df.Sex = np.where(df["Sex"] == 'male', 0, 1) 					 # Change "Sex" to numeric values

	df.Pclass = df.Pclass / 3 					                     # Shorten Pclass to values between 1.0 and 0.0

	df.Embarked = df.loc[df["Embarked"].isnull(), "Embarked"] = 0

	max_age = max(df.Age)
	avg_age = np.average(df.dropna().Age)

	df.loc[df["Age"] < 1.0, "Age"] = 1.001      					 # set ages less than 1 to 1.001
	df.Age = df.Age.fillna(avg_age)                                  # fills NaN entries for age with log average
	df.Age = np.log(df.Age) / np.log(max_age)                        # squishes the age to log weighted average between 0 and 1

	max_fare = max(df.Fare)											 # get maxiumum fare
	df.Fare = np.log(df.Fare) / np.log(max_fare)					 # sets the fare to log weighted average between 0 and 1
	df.loc[df["Fare"] < 0, "Fare"] = 0								 # sets fare prices of -inf to 0

	return df



class TorchDataset(Dataset):
	def __init__(self, filepath, transform=None, target_transform=None):
		df = pd.read_csv(filepath)
		df = df.drop(["Unnamed: 0"], axis=1) # drop unnamed axis

		df1 = df.copy()
		not_survived = 1 - df1.Survived
		labels = pd.DataFrame(data = {"survived":df1.Survived, "not_survived":not_survived})
		self.labels = Tensor(labels.to_numpy())

		self.data = Tensor(df.drop(["Survived"], axis = 1).to_numpy())

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):    
		inp = self.data[idx]
		tar = self.labels[idx]
		if self.transform:
			inp = self.transform(inp)
		if self.target_transform:
			tar = self.target_transform(tar)
		return inp, tar

class TorchOutDataset(Dataset):
	def __init__(self, filepath, transform=None, target_transform=None):
		df = pd.read_csv(filepath)
		df = df.drop(["Unnamed: 0"], axis=1) # drop unnamed axis

		self.data = Tensor(df.to_numpy())

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):    
		inp = self.data[idx]
		return inp, Tensor([0, 0])
