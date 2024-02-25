import torch
import torch.nn as nn
import torch.nn.functional as F

# 정확도를 높이기위한 loss term 추가.
class myLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=0.25):
        super(myLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-9

    def forward(self, y_true, predict):
        # Check that predictions are in the range [0, 1]
        if not (0 <= predict).all() and (predict <= 1).all():
            raise ValueError("Predictions should be in the range [0, 1].")

#         loss = -torch.mean(self.alpha * y_true * torch.log(predict + self.eps) + ((1 - predict + self.eps) ** self.gamma) * (1 - y_true) * torch.log(1 - predict + self.eps))

#         loss = -torch.mean( self.alpha*y_true * torch.log(predict + self.eps) + (1 - y_true) * torch.log(1 - predict + self.eps))
        loss = -torch.mean(y_true * torch.log(predict + self.eps) +(1 - y_true) * torch.log(1 - predict + self.eps))
        return loss

if __name__=="__main__":
    loss = myLoss()
    x = torch.exp(torch.randn(77))
    x= x/x.sum()
    y = torch.exp(torch.randn(77))
    y = y/y.sum()
    print(x,y)
    loss1 = loss(x,y)
    print(loss1)