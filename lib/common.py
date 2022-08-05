import torch
import torch.backends.cudnn
from torch import nn

from lib.Dice_loss import dice_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def criterion(inputs, target, dice: bool = False):
    loss1 = 0
    if dice:
        loss1 = dice_loss(inputs, target)
    target = target.unsqueeze(1).float()
    weight = torch.ones_like(target).float() * 2
    weight[target == 0] = 1
    loss2 = nn.BCELoss(weight=weight)(inputs, target)
    return loss1 + loss2


# Round off
def dict_round(dic, num):
    for key, value in dic.items():
        dic[key] = round(value, num)
    return dic


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
