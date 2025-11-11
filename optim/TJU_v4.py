import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import math


class TJU_v4(Optimizer):
    r"""
    Fixed TJU_AdamW optimizer, maintaining the original TJU_v3 precision while implementing correct decoupled weight decay

    Key improvements:
    1. Fix the weight decay application: scale with current learning rate (current_lr * weight_decay)
    2. Maintain the original TJU_v3 approximate Hessian processing logic
    3. Restore the original parameter update order to ensure numerical stability
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            beta_h=0.85,
            eps=1e-8,
            rebound='constant',
            warmup=100,
            init_lr=None,
            weight_decay=0.0,
            weight_decay_type='L2',
            hessian_scale=0.05,
            total_steps=10000,
            use_cosine_scheduler=True
    ):
        # Parameter validation (maintain original strict validation)
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:  # Correct the options list
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta_h=beta_h,
            eps=eps,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr or lr / 1000.0,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_type=weight_decay_type,
            hessian_scale=hessian_scale,
            total_steps=total_steps,
            use_cosine_scheduler=use_cosine_scheduler
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TJU_AdamW_Fixed does not support sparse gradients")

                state = self.state[p]
                # Initialize state (maintain original TJU_v3 structure)
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # ====== Learning rate scheduling (maintain original TJU_v3 logic) ======
                current_lr = self._compute_lr(group, step)

                # ====== Core parameter update ======
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                # (1) L2 regularization (maintain original logic)
                if group['weight_decay_type'] == 'L2' and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # (2) Update momentum terms (maintain original TJU_v3 numerical stability)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # (3) Bias correction (key! Maintain original TJU_v3 implementation)
                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1  # Combine learning rate with first-order bias correction

                # (4) Approximate Hessian processing (maintain original TJU_v3 clamp logic)
                delta_grad = grad - (exp_avg / bias_corr1)  # Corrected gradient change amount
                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)  # Maintain original v3 clamp lower bound
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                # (5) Combine second moments (maintain original v3 mixing logic)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['eps'])

                # (6) Calculate update direction (key modification! Restore original v3 stability)
                update = exp_avg / denom

                # (7) Handle stable type weight decay (maintain original v3 logic)
                if group['weight_decay_type'] == 'stable' and group['weight_decay'] != 0:
                    decay_factor = group['weight_decay'] / denom.mean().clamp(min=1e-8)
                    update.add_(p, alpha=decay_factor)

                # ====== AdamW type weight decay (key fix!) ====== #
                # Apply decoupled decay during parameter update (keep it independent of current learning rate)
                if group['weight_decay_type'] == 'AdamW' and group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'] * current_lr)  # Key modification for decoupling with learning rate!

                # (8) Perform parameter update (maintain original v3 update order)
                p.add_(update, alpha=-step_size)  # Note: step_size already includes learning rate and first-order bias correction

        return loss

    def _compute_lr(self, group, step):
        """Learning rate scheduling (precisely maintain original TJU_v3 implementation)"""
        if step <= group['warmup']:
            return group['init_lr'] + (group['base_lr'] - group['init_lr']) * step / group['warmup']

        if not group['use_cosine_scheduler']:
            return group['base_lr']

        t = step - group['warmup']
        T = group['total_steps'] - group['warmup']
        if t <= T:
            return group['base_lr'] * (0.5 * (1 + math.cos(math.pi * t / T)))
        return group['base_lr'] * 0.01  # Maintain original v3 post-training phase learning rate