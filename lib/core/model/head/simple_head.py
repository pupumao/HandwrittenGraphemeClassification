
import torch.nn as nn
import torch
class Header(nn.Module):

    def __init__(self, input_dims,dropout_p=None):
        super(Header, self).__init__()


        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None



        # self.last_1 = nn.Linear(input_dims, 168)
        # self.last_2 = nn.Linear(input_dims, 11)
        # self.last_3 = nn.Linear(input_dims, 7)

        self.last_1 = nn.Conv2d(input_dims, 168,(1,1),1,bi)
        self.last_2 = nn.Conv2d(input_dims, 11,(1,1),1)
        self.last_3 = nn.Conv2d(input_dims, 7,(1,1),1)

    def logits(self, x):


        x = self.avg_pool(x)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        x1 = self.last_1(x)
        x2 = self.last_1(x)
        x3 = self.last_1(x)
        x1 = torch.squeeze(x1,dim=-1)
        x2 = torch.squeeze(x2, dim=-1)
        x3 = torch.squeeze(x3, dim=-1)
        return x1,x2,x3

    def forward(self, x):

        x1,x2,x3 = self.logits(x)
        return x1,x2,x3
