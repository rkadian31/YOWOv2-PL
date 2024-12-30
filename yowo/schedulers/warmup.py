from typing import List, Literal
import warnings

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from yowo.utils.validate import validate_literal_types


class WarmupLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        name: Literal["exp", "linear", "cosine"] = 'linear',
        max_iter: int = 500,
        factor: float = 0.00066667,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
        high_resolution: bool = False,
        resolution_scale: float = 1.0,
        warmup_momentum: float = 0.9,
        warmup_bias_lr: Optional[float] = None
    ):
        """
        Enhanced warmup scheduler with high resolution support.

        Args:
            optimizer (Optimizer): The optimizer to use
            name (Literal["exp", "linear", "cosine"]): Warmup schedule type
            max_iter (int): Maximum warmup iterations
            factor (float): Base warmup factor
            last_epoch (int): Index of last epoch
            min_lr (float): Minimum learning rate
            high_resolution (bool): Enable high resolution mode
            resolution_scale (float): Scale factor for high resolution
            warmup_momentum (float): Momentum during warmup
            warmup_bias_lr (Optional[float]): Separate learning rate for bias terms
        """
        validate_literal_types(name, Literal["exp", "linear", "cosine"])
        self.name = name
        self.min_lr = min_lr
        self.high_resolution = high_resolution
        self.resolution_scale = resolution_scale
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr

        # Adjust parameters for high resolution
        if high_resolution:
            self.max_iter = int(max_iter * resolution_scale)
            self.factor = factor * resolution_scale
        else:
            self.max_iter = max_iter
            self.factor = factor

        super().__init__(optimizer, last_epoch)

        # Initialize momentum buffer
        if self.high_resolution:
            for group in optimizer.param_groups:
                group['momentum'] = self.warmup_momentum

    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch < self.max_iter:
            tmp_lrs = self.warmup(iter=self.last_epoch)
            ratios = [
                group['initial_lr'] / base_lr 
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)
            ]
            
            # Apply separate learning rate for bias terms if specified
            if self.warmup_bias_lr is not None:
                final_lrs = []
                for tmp_lr, ratio, group in zip(tmp_lrs, ratios, self.optimizer.param_groups):
                    if 'bias' in group['name']:
                        final_lrs.append(self.warmup_bias_lr * ratio)
                    else:
                        final_lrs.append(tmp_lr * ratio)
                return final_lrs
            
            return [max(tmp_lr * ratio, self.min_lr) for tmp_lr, ratio in zip(tmp_lrs, ratios)]
            
        elif self.last_epoch == self.max_iter:
            return [max(group['initial_lr'], self.min_lr) for group in self.optimizer.param_groups]
        else:
            return [max(group['lr'], self.min_lr) for group in self.optimizer.param_groups]

    def warmup(self, iter: int):
        """Enhanced warmup with multiple strategies"""
        if self.name == 'exp':
            # Modified exponential warmup for high resolution
            power = 4 if not self.high_resolution else 3
            tmp_lrs = [
                base_lr * pow(iter / self.max_iter, power)
                for base_lr in self.base_lrs
            ]

        elif self.name == 'linear':
            # Enhanced linear warmup with smoothing
            alpha = iter / self.max_iter
            if self.high_resolution:
                # Smoother transition for high resolution
                alpha = self._smooth_transition(alpha)
            warmup_factor = self.factor * (1 - alpha) + alpha
            tmp_lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif self.name == 'cosine':
            # Cosine warmup schedule
            alpha = iter / self.max_iter
            if self.high_resolution:
                alpha = self._smooth_transition(alpha)
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - alpha)))
            tmp_lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Apply resolution-specific adjustments
        if self.high_resolution:
            tmp_lrs = self._adjust_for_resolution(tmp_lrs, iter)

        return [max(lr, self.min_lr) for lr in tmp_lrs]

    def _smooth_transition(self, alpha: float) -> float:
        """Smooth transition function for high resolution training"""
        return 0.5 * (1 + math.tanh(10 * (alpha - 0.5)))

    def _adjust_for_resolution(self, lrs: List[float], iter: int) -> List[float]:
        """Apply resolution-specific adjustments to learning rates"""
        if iter < self.max_iter * 0.2:
            # Slower warmup initially for high resolution
            scale = math.pow(iter / (self.max_iter * 0.2), 2)
        else:
            # Gradual scaling based on resolution
            scale = self.resolution_scale

        return [lr * scale for lr in lrs]

    def state_dict(self) -> dict:
        """Enhanced state dict with additional parameters"""
        state_dict = super().state_dict()
        state_dict.update({
            'high_resolution': self.high_resolution,
            'resolution_scale': self.resolution_scale,
            'warmup_momentum': self.warmup_momentum,
            'min_lr': self.min_lr
        })
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Enhanced state dict loading with additional parameters"""
        self.high_resolution = state_dict.pop('high_resolution', False)
        self.resolution_scale = state_dict.pop('resolution_scale', 1.0)
        self.warmup_momentum = state_dict.pop('warmup_momentum', 0.9)
        self.min_lr = state_dict.pop('min_lr', 1e-6)
        super().load_state_dict(state_dict)

# class WarmupLR(Callback):
#     def __init__(
#         self,
#         name: str = 'linear',
#         base_lr: float = 0.01,
#         max_iteration: int = 500,
#         warmup_factor: float = 0.00066667
#     ):
#         super().__init__()
#         self.name = name
#         self.base_lr = base_lr
#         self.max_iteration = max_iteration
#         self.warmup_factor = warmup_factor
#         self.warmup = True

#     def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
#         if stage == "fit":
#             self.warmup_scheduler = WarmUpScheduler(
#                 name=self.name,
#                 base_lr=self.base_lr,
#                 wp_iter=self.max_iteration,
#                 warmup_factor=self.warmup_factor
#             )

#     def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
#         opt = pl_module.optimizers()
#         if pl_module.global_step < self.max_iteration and self.warmup:
#             self.warmup_scheduler.warmup(
#                 iter=pl_module.global_step,
#                 optimizer=opt
#             )
#         elif pl_module.global_step >= self.max_iteration and self.warmup:
#             self.warmup = False
#             self.warmup_scheduler.set_lr(
#                 optimizer=opt,
#                 lr=self.base_lr,
#                 base_lr=self.base_lr
#             )

#     def state_dict(self) -> Dict[str, Any]:
#         return {key: value for key, value in self.__dict__.items()}

#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         self.__dict__.update(state_dict)
