import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, y_shot, x_query, y_query):
        x_all = torch.cat([x_shot, x_query], dim=0)
        x_all = self.encoder(x_all)
        x_shot, x_query = x_all[:len(x_shot)], x_all[-len(x_query):]

        n_way = int(y_shot.max()) + 1
        proto = []
        for c in range(n_way):
            ind = []
            for i, y in enumerate(y_shot):
                if int(y) == c:
                    ind.append(i)
            proto.append(x_shot[ind].mean(dim=0))
        proto = torch.stack(proto)

        logits = utils.compute_logits(x_query, proto, metric='cos', temp=self.temp)
        loss = F.cross_entropy(logits, y_query)
        acc = utils.compute_acc(logits, y_query)

        return loss, acc
