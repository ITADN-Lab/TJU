import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class TJU_v1(Optimizer):
    r"""
    TJU_v1: A variant of Atom/TJU-based optimizer for deep learning.

    This optimizer integrates approximate Hessian information with an EMA of gradients
    to guide parameter updates. It supports various weight decay strategies and warmup
    schedules for more flexible training.

    Args:
        params (iterable):
            Model parameters to optimize (can be a single iterable or multiple param groups).
        lr (float):
            Base learning rate for updates.
        beta (float, optional):
            Momentum factor for gradient EMA. Default: 0.9
        eps (float, optional):
            A small constant for numerical stability (denominator addition). Default: 1e-4
        rebound (str, optional):
            Mode for bounding the diagonal Hessian. {'constant', 'belief'}. Default: 'constant'
        warmup (int, optional):
            Number of warmup steps during which the learning rate ramps from init_lr to lr. Default: 500
        init_lr (float, optional):
            Learning rate used at start of warmup. Default: lr/1000
        weight_decay (float, optional):
            Weight decay coefficient. Default: 0
        weight_decay_type (str, optional):
            Weight decay type: {'L2', 'decoupled', 'stable'}.
            If unset, defaults to 'L2' for 'constant' rebound or 'decoupled' otherwise.
    """

    def __init__(
            self,
            params,
            lr,
            beta=0.9,
            eps=1e-4,
            rebound='constant',
            warmup=500,
            init_lr=None,
            weight_decay=0,
            weight_decay_type=None
    ):
        # -- Parameter validity checks --
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate value: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta} (must be in [0.0, 1.0))")
        if rebound not in ['constant', 'belief']:
            raise ValueError(f"Invalid rebound mode: {rebound}, must be 'constant' or 'belief'")
        if not 0 <= warmup:
            raise ValueError(f"Invalid warmup steps: {warmup} (must be >= 0)")

        # If init_lr is not given, default to init_lr = lr / 1000
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError(f"Invalid init_lr: {init_lr} (must be in [0, lr])")

        # Weight decay coefficient and type
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay} (must be >= 0)")
        if weight_decay_type is None:
            # If rebound='constant', default to L2, otherwise decoupled
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type} "
                             "(must be 'L2', 'decoupled', or 'stable')")

        # Use a dictionary to store default hyperparameters uniformly
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type
        )
        super(TJU_v1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TJU_v1, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.

        Returns:
            loss (float, optional): The loss from closure if provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate through each param group in sequence
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            rebound_mode = group['rebound']
            warmup_steps = group['warmup']
            init_lr = group['init_lr']
            base_lr = group['base_lr']
            wd_coef = group['weight_decay']
            wd_type = group['weight_decay_type']

            # Iterate through the parameters in this group
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get the state dictionary of the current parameter
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of first-order gradients
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of approximate Hessian
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous update direction
                    state['prev_update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                step_count = state['step']
                if step_count < warmup_steps:
                    # Linearly ramp from init_lr to base_lr
                    current_lr = (base_lr - init_lr) * (step_count / warmup_steps) + init_lr
                else:
                    current_lr = lr

                step_count += 1
                state['step'] = step_count

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TJU_v3 does not support sparse gradients.")

                # If L2 decay, add the decay term directly to the gradient
                if wd_coef != 0 and wd_type == 'L2':
                    grad = grad.add(p, alpha=wd_coef)

                exp_avg_grad = state['exp_avg_grad']
                approx_hessian = state['approx_hessian']
                prev_update = state['prev_update']

                bias_corr = 1 - (beta ** step_count)
                effective_alpha = (1 - beta) / bias_corr

                # Compute gradient difference
                grad_diff = grad - exp_avg_grad

                # Select threshold based on rebound mode
                if rebound_mode == 'belief':
                    rebound_thresh = grad_diff.norm(p=np.inf)
                else:
                    rebound_thresh = 0.01
                    eps = eps / max(rebound_thresh, 1e-8)  # Prevent division by 0

                # (1) Update first-order gradient average: exp_avg_grad ← exp_avg_grad + effective_alpha * grad_diff
                exp_avg_grad.add_(grad_diff, alpha=effective_alpha)

                # (2) Normalize the previous update for computing the update to the approximate Hessian
                prev_update_norm = prev_update.norm(p=4).add(eps)
                prev_update.div_(prev_update_norm)
                prev_update_sq = prev_update.mul(prev_update)

                # Compute delta for updating Hessian
                # Inner product of (grad_diff / prev_update_norm) with prev_update
                delta_term = (grad_diff.div_(prev_update_norm)
                              .mul_(prev_update)
                              .sum()
                              .mul_(-effective_alpha)
                              ) - approx_hessian.mul(prev_update_sq).sum()

                # (3) Update Hessian: approx_hessian ← approx_hessian + (delta_term * prev_update_sq)
                approx_hessian.addcmul_(prev_update_sq, delta_term)

                # (4) Compute the new update direction new_update
                if rebound_mode == 'belief':
                    denom_h = torch.max(approx_hessian.abs(), rebound_thresh)
                    denom_h.add_(eps / effective_alpha)
                else:
                    denom_h = approx_hessian.abs().clamp_(min=rebound_thresh)

                new_update = exp_avg_grad.div(denom_h)

                # (5) If decoupled or stable wd is needed
                if wd_coef != 0 and wd_type != 'L2':
                    if wd_type == 'stable':
                        scaled_decay = wd_coef / max(denom_h.mean().item(), 1e-8)
                        new_update.add_(p, alpha=scaled_decay)
                    else:
                        # decoupled
                        new_update.add_(p, alpha=wd_coef)

                # (6) Update parameters
                p.add_(new_update, alpha=-current_lr)

                # Store the current update vector for use in the next calculation
                prev_update.copy_(new_update)

        return loss