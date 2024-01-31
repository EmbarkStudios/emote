import torch
from torch import Tensor


def gradient_loss_function(prediction: Tensor, inputs: Tensor) -> Tensor:
    imitation_batch_size = prediction.shape[0]
    n_obs = inputs.shape[1]
    predictions = torch.split(prediction, 1, dim=0)
    grad_inp = torch.autograd.grad(predictions, inputs, create_graph=True, retain_graph=True)
    grad_inp = torch.cat(grad_inp, axis=1)
    assert grad_inp.shape == (imitation_batch_size, n_obs)
    grad_inp_norm = torch.square(grad_inp)
    assert grad_inp_norm.shape == (imitation_batch_size, n_obs)
    grad_inp_norm = torch.sum(grad_inp_norm, dim=1)
    assert grad_inp_norm.shape == (imitation_batch_size,)
    grad_inp_loss = torch.mean(grad_inp_norm)
    assert grad_inp_loss.shape == ()
    return grad_inp_loss


batch_size = 10
feature_size = 3
a = torch.ones(batch_size, feature_size, requires_grad=True) * 3
b = torch.ones(batch_size, feature_size, requires_grad=True) * 2

model_input = torch.cat((2 * a * a, 2 * b), dim=1)
print("model input: ", model_input)
model_output = torch.sum(model_input, dim=1)

print("model output: ", model_output)

loss = gradient_loss_function(model_output, a)
print("gradient loss: ", loss)