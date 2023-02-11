from torch import nn
import torch


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss,self).__init__()

        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x)) / torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = 1
    
    def forward(self, inp, target, test=False):
        assert(inp.shape == target.shape)        

        if inp.is_cuda:
            self.prior = self.prior.cuda()

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        n_positive, n_unlabeled = torch.clamp(torch.sum(positive), min=self.min_count), torch.clamp(torch.sum(unlabeled), min=self.min_count)

        y_positive = self.loss_func(inp) * positive
        y_positive_inv = self.loss_func(-inp) * positive
        y_unlabeled = self.loss_func(-inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk

        return positive_risk + negative_risk
