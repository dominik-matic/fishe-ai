import numpy as np
import torch

PRINT_PERIOD = 10

def dense_to_one_hot(label, class_count):
	return torch.tensor(np.eye(class_count)[label])

def train(model, optimizer, loader, device='cuda'):
	losses = []
	model.train()
	for i, data in enumerate(loader):
		img, label = data
		label_oh = dense_to_one_hot(label, 442)
		optimizer.zero_grad()
		#loss = model.loss(img.to(device), label_oh.to(device))
		
		loss = torch.nn.CrossEntropyLoss(reduction='mean')(model(img.to(device))[0], label_oh.to(device)) # inception_v3 only
		#loss = torch.nn.CrossEntropyLoss(reduction='mean')(model(img.to(device)), label_oh.to(device))
		
		loss.backward()
		optimizer.step()
		losses.append(loss.cpu().item())
		if i % PRINT_PERIOD == 0:
			print(f'Iter {i}: mean loss: {np.mean(loss.cpu().item()):.3f}')
	return np.mean(losses)

def evaluate(model, loader, device='cuda'):
	model.eval()
	total = 0
	correct = 0
	for i, data in enumerate(loader):
		img, label = data
		with torch.no_grad():

			#logits = model.forward(img.to(device))
			logits = model.forward(img.to(device)[0]) # inception_v3 only
			logits = torch.exp(logits)
			sumexp = torch.sum(logits, axis=1)
			predicted = logits / sumexp[:, None]
			predicted_label = torch.argmax(predicted, dim=1)
			for true_label, predicted_label in zip(label.cpu(), predicted_label.cpu()):
				total += 1
				if true_label == predicted_label:
					correct += 1
	return correct / total