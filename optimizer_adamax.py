import torch
from torch.optim import Optimizer

class Adamax(Optimizer):
    def __init__(
            self,
            params,
            lr=0.002,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            correct_bias=True
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(Adamax, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adamax does not support sparse gradients, please consider SparseAdamax instead")

                state = self.state[p]

                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                if 'step' not in state:
                    state['step'] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["u"] = torch.zeros_like(p.data)
                    if group["correct_bias"]:
                        state["bias_correction"] = 1 - beta1 ** state["step"]
                else:
                    state["bias_correction"] = 1 - beta1 ** state["step"]

                state['step'] += 1

                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["u"] = torch.max(beta2 * state["u"], torch.abs(grad))

                m_hat = state["m"] / (1 - beta1 ** state["step"])
                step_size = group['lr'] / (state["u"] + eps)

                if weight_decay != 0:
                    p.data = p.data - group['lr'] * weight_decay * p.data

                p.data = p.data - step_size * m_hat

        return loss
