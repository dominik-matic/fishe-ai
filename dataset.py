from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch import rand
import torchvision
import torchvision.transforms as transforms


def get_preprocess_transform():
	return transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
def get_train_transform():
	return transforms.Compose([
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomAffine(degrees=(0, 360), translate=(0, 0.3), scale=(0.5, 1.5))
		])

def make_dataset(root='./data/NA_Fish_Dataset/NA_Fish_Dataset_Modified', transform=get_preprocess_transform()):
	return torchvision.datasets.ImageFolder(root=root, transform=transform)

class FisheTrainDataset(Dataset):
	def __init__(self, dataset, train_transform=get_train_transform(), train_transform_p = 0.9):
		self.dataset = dataset
		self.train_transform = train_transform
		self.train_transform_p = train_transform_p
	
	def __getitem__(self, index):
		sample, target = self.dataset[index]
		if rand(1).item() < self.train_transform_p:
			return self.train_transform(sample), target
		return sample, target
	
	def __len__(self):
		return len(self.dataset)

def main():
	dataset = torchvision.datasets.ImageFolder(root='./data/Adriatic_Fish_Dataset/AFD_Unmodified')
	for i, (img, label) in enumerate(dataset):
		im = img
		for j in range(50):
			im.save(f'./data/Adriatic_Fish_Dataset/AFD_Modified/{dataset.classes[label]}/{i}{j}.jpg')
			im = __train_transform(img)

def main2():
	ds = make_dataset()
	ds_size = len(ds)
	ds_train, ds_test, ds_validation = random_split(ds, [int(0.6*ds_size), int(0.2*ds_size), int(0.2*ds_size)])
	ds_train = FisheTrainDataset(ds_train)
	print(len(ds_train))
	print(len(ds_test))
	print(len(ds_validation))
	print(ds_train[0])
	

if __name__ == '__main__':
	main2()