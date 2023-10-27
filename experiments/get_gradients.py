import torch
from torch import Tensor


def grad_penalty_loss(
        prior_pred: Tensor, prior_obs: Tensor, next_prior_obs: Tensor
) -> Tensor:
    prior_bsz = prior_pred.shape[0]
    n_obs = prior_obs.shape[1]

    prior_preds = torch.split(prior_pred, 1, dim=0)

    print("prior_preds:", prior_preds)
    # this calculates the gradient of the discriminator output w.r.t the prior input.
    grad_inp = torch.autograd.grad(
        prior_preds, [prior_obs, next_prior_obs], create_graph=True, retain_graph=True
    )

    # grad inp is now a tuple of ([bsz, n_obs], [bsz, n_obs])
    # first is the prior_obs grad, 2nd is the next_prior_obs grad
    # we cat together the obs input.
    grad_inp = torch.cat(grad_inp, axis=1)
    assert grad_inp.shape == (prior_bsz, 2 * n_obs)

    print("*" * 10)
    print("grad input:", grad_inp)
    print("*" * 10)

    # square each elem
    grad_inp_norm = torch.square(grad_inp)
    assert grad_inp_norm.shape == (prior_bsz, 2 * n_obs)

    # sum each squared elem (l2 norm squared)
    grad_inp_norm = torch.sum(grad_inp_norm, dim=1)
    print("grad_inp_norm:", grad_inp_norm)
    print("*" * 10)

    assert grad_inp_norm.shape == (prior_bsz,)

    # reduce mean
    grad_inp_loss = torch.mean(grad_inp_norm)
    assert grad_inp_loss.shape == ()  # this should be a scalar.

    return grad_inp_loss


batch_size = 10
feature_size = 3
a = torch.ones(batch_size, feature_size, requires_grad=True)
b = torch.ones(batch_size, feature_size, requires_grad=True)

pred_input = torch.cat((3*a, 2*b), dim=1)
print(pred_input)
pred = torch.sum(pred_input, dim=1)

print(pred_input.shape)
print(pred.shape)

loss = grad_penalty_loss(pred, a, b)
print(loss)