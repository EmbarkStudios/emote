import torch
from emote.algorithms.amp import gradient_loss_function


def test_gradient_loss():

    x = torch.ones(10, 3, requires_grad=True)
    x = x * torch.rand(10, 3)
    y = torch.sum(4 * x * x + torch.sin(x), dim=1)

    grad1 = gradient_loss_function(y, x)
    y_dot = 8 * x + torch.cos(x)
    grad2 = torch.mean(torch.sum(y_dot * y_dot, dim=1))

    assert grad1.item() == grad2.item()
