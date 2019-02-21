from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam

from .model import PUModel
from .loss import PULoss
from .dataset import PU_MNIST, PN_MNIST


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_loader, optimizer, prior, epoch):
    model.train()
    tr_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss_fct = PULoss(prior = prior)
        loss = loss_fct(output.view(-1), target.type(torch.float))
        tr_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print("Train loss: ", tr_loss)


def test(args, model, device, test_loader, prior):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0
    num_pos = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss_func = PULoss(prior = prior)
            test_loss += test_loss_func(output.view(-1), target.type(torch.float)).item() # sum up batch loss
            pred = torch.where(output < 0, torch.tensor(-1, device=device), torch.tensor(1, device=device)) 
            num_pos += torch.sum(pred == 1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Percent of examples predicted positive: ', float(num_pos)/len(test_loader.dataset), '\n')


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

   
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--nnPU",
                        action='store_true',
                        help="Whether to us non-negative pu-learning risk estimator.")
    parser.add_argument("--train_batch_size",
                        default=30000,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=100,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_set = PU_MNIST(args.data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
    prior = train_set.get_prior()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    test_set = PN_MNIST(args.data_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True, **kwargs)

    model = PUModel().to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.005)

    if args.do_train:
        for epoch in range(1, args.num_train_epochs + 1):
            train(args, model, device, train_loader, optimizer, prior, epoch)
            test(args, model, device, test_loader, prior)

        torch.save(model.state_dict(), output_model_file)

    elif args.do_eval:
        test(args, model, device, test_loader, prior)

if __name__ == "__main__":
    main()
