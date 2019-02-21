import torch
from torch.utils import data
from torchvision.datasets import MNIST


class PU_MNIST(MNIST):
	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(PU_MNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
		self.pos_indices = []
		i = 0
		while len(self.pos_indices) < 1000:
			if self.train_labels[i] % 2 == 0:
				self.pos_indices.append(i)
			i+=1


	def __getitem__(self, i):
	    input, target = super(PU_MNIST, self).__getitem__(i)
	    if i in self.pos_indices:
	    	target = torch.tensor(1)
	    else:
	    	target = torch.tensor(-1)
	      
	    return input, target

	def get_prior(self):
		pos_examples = self.train_labels % 2 == 0
		return torch.sum(pos_examples, dtype=torch.float)/len(self.train_labels)



class PN_MNIST(MNIST):
	def __getitem__(self, i):
	    input, target = super(PN_MNIST, self).__getitem__(i)
	    if target % 2 == 0:
	    	target = torch.tensor(1)
	    else:
	    	target = torch.tensor(-1)
	      
	    return input, target