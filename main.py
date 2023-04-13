import torch
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from copy import deepcopy
from torch.utils.data import DataLoader
from model import DeepFisheNetV7
from dataset import make_dataset
from dataset import FisheTrainDataset
from utils import train
from utils import evaluate

#from torchvision.models import resnet50
from torchvision.models import inception_v3



def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f'Device: {device}')

	custom_transform = transforms.Compose([
		transforms.Resize([299, 299]), # inception needs 299, 299
		transforms.ToTensor(),
		#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	ds = make_dataset(root='./data/Adriatic_Fish_Dataset/AFD_Unmodified', transform=custom_transform) # downscaling
	#ds = make_dataset(root='./data/Adriatic_Fish_Dataset/AFD_Unmodified') # no downscaling
	ds_size = len(ds)


	ds_train, ds_test, ds_validation = torch.utils.data.random_split(ds, [round(0.8*ds_size) + 1, round(0.1*ds_size), round(0.1*ds_size)], torch.Generator().manual_seed(71))
	ds_train = FisheTrainDataset(ds_train)


	print(f'Loaded {len(ds_train)} training images')
	print(f'Loaded {len(ds_validation)} validation images')
	print(f'Loaded {len(ds_test)} test images')

	bsize = 4
	train_loader = DataLoader(
		ds_train,
		batch_size=bsize,
		shuffle=True,
		pin_memory=True,
		num_workers=4,
		drop_last=False
	)

	test_loader = DataLoader(
		ds_test,
		batch_size=bsize,
		shuffle=False,
		pin_memory=True,
		num_workers=1,
		drop_last=False
	)

	validation_loader = DataLoader(
		ds_validation,
		batch_size=bsize,
		shuffle=False,
		pin_memory=True,
		num_workers=1,
		drop_last=False
	)


	#model = resnet50(pretrained=True)
	model = inception_v3(pretrained=True, progress=False)
	model.fc = torch.nn.Linear(2048, 442)
	
	
	model.to(device)

	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=1e-4,
		weight_decay=1e-4
	)

	epochs = 300
	best_acc = 0
	best_model = deepcopy(model.state_dict())
	accs = []
	t_0 = time.time_ns()
	repeat = True
	while repeat:
		for epoch in range(1, epochs + 1):
			te_0 = time.time_ns()
			print(f'Epoch: {epoch}')
			train_loss = train(model, optimizer, train_loader, device)
			print(f'Mean loss in epoch {epoch}: {train_loss:.3f}')
			print(f'Evaluating on validation set...')
			acc = evaluate(model, validation_loader, device)
			if acc > best_acc:
				best_acc = acc
				best_model = deepcopy(model.state_dict())
			accs.append(acc)
			print(f'Epoch {epoch} validation accuracy: {acc * 100:.2f}%')
			te_1 = time.time_ns()
			print(f'Epoch time: {(te_1 - te_0) / 10**9:.1f}s')
			if epoch % 10 == 0:
				print('Evaluating on train set...')
				train_acc = evaluate(model, train_loader)
				print(f'Train accuracy: {train_acc * 100:.2f}%')
		yn = ''
		while yn not in ['y', 'n']:
			yn = input('Do you want to continue training? (y/n) ').lower()
			if yn == 'y':
				new_epochs = ''
				while not new_epochs.isnumeric():
					new_epochs = input('How many epochs: ')
				epochs = int(new_epochs)
			elif yn == 'n':
				repeat = False
	print(f'Best validation accuracy: {best_acc * 100:.2f}%')
	print(f'Evaluating on test set on best model based on best validation accuracy...')
	model.load_state_dict(best_model)
	acc = evaluate(model, test_loader, device)
	print(f'Test accuracy: {acc * 100:.2f}%')
	t_1 = time.time_ns()
	print(f'Total time: {(t_1 - t_0) / 10**9:.1f}s')
	
	plt.plot(accs)
	plt.show()
	
	yn = ''
	while yn not in ['y', 'n']:
		yn = input('Do you want to save this model? (y/n) ').lower()
		if yn == 'y':
			print('Saving model...')
			torch.save(best_model, './saved_models/model_acc_' + str(acc))
			print('Done.')
	
	

if __name__ == '__main__':
	main()