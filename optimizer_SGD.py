import torch
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(
            self,
            params,
            lr=0.01,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {} - should be >= 0.0".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {} - should be >= 0.0".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

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
                    raise RuntimeError("SGD does not support sparse gradients")

                state = self.state[p]

                d_p = grad

                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)

                if group['momentum'] != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(1 - group['dampening'], d_p)
                    if group['nesterov']:
                        d_p = d_p.add(group['momentum'], buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
