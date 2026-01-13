import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
from losses.loss_other import em_weight

class Mixup(nn.Module):

    def __init__(self, gpu=None, num_classes=10, alpha=5.0, model=None):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device('cuda:%s' %
                                   gpu) if gpu else torch.device('cpu')
        self.alpha = torch.tensor(alpha).to(self.device)
        if model:
            model.dummy_head = nn.Linear(512, num_classes,
                                         bias=True).to(self.device)
            model.fc.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, data_in, data_out):
        self.features = data_in[0]

    def forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = torch.zeros(b, self.num_classes).to(self.device).scatter_(
            1, y.view(-1, 1), 1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p,_ = model(mixed_x)
        loss = -torch.mean(
            torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss
    
    def h_forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        output = model(mixed_x) #batchsize*hashbits
        if(isinstance(output,tuple)):
            _,mixed_output = output
        else:
            mixed_output = output
        criterion = nn.BCELoss(reduction="none").to(self.device)
        loss = torch.mean(criterion(0.5*(mixed_output+1),0.5*(mixed_y+1)))
        return loss
    
    def we_h_forward(self, x, y, model,lam):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_output = model(mixed_x)     #batchsize*hashbits
        criterion = nn.BCELoss(reduction="none").to(self.device)
        losses = criterion(0.5*(mixed_output+1),0.5*(mixed_y+1))
        with torch.no_grad():
            weight = em_weight(losses,lam)
        loss = torch.mean(weight*torch.mean(losses,dim=1),dim=0) 
        return loss
    




    def ws_forward(self, wx, sx, y, model):
        b = wx.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = torch.zeros(b, self.num_classes).to(self.device).scatter_(
            1, y.view(-1, 1), 1)
        mixed_x = lam * wx + (1 - lam) * sx[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(
            torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss


    def soft_forward(self, x, p, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        index = torch.randperm(b).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * p + (1 - lam) * p[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(
            torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def dummy_forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = torch.zeros(b, self.num_classes).to(self.device).scatter_(
            1, y.view(-1, 1), 1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        mixed_features = self.features.clone()
        dummy_logits = model.dummy_head(mixed_features)
        loss = -torch.mean(
            torch.sum(F.log_softmax(dummy_logits, dim=1) * mixed_y, dim=1))
        return loss