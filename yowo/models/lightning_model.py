from dataclasses import asdict
from typing import Any, Literal, Mapping

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from yowo.utils.box_ops import rescale_bboxes_tensor
from .yowov2.model import YOWO
from .yowov2.loss import build_criterion

from .schemas import (
    LossConfig,
    ModelConfig,
    LRSChedulerConfig
)


class YOWOv2Lightning(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        optimizer: OptimizerCallable,
        scheduler_config: LRSChedulerConfig,
        warmup_config: LRSChedulerConfig | None,
        freeze_backbone_2d: bool = True,
        freeze_backbone_3d: bool = True,
        metric_iou_thresholds: list[float] | None = [0.25, 0.5, 0.75, 0.95],
        metric_rec_thresholds: list[float] | None = [0.1, 0.3, 0.5, 0.7, 0.9],
        metric_max_detection_thresholds: list[int] | None = [1, 10, 100],
        high_resolution: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Add high resolution handling
        self.high_resolution = high_resolution or (
            model_config.img_size[0] >= 1080 or 
            model_config.img_size[1] >= 1920
        )
        
        # Adjust model configuration for high resolution
        if self.high_resolution:
            model_config.stride = [s * 2 for s in model_config.stride]
            model_config.head_dim *= 2
            loss_config.center_sampling_radius *= 2
            loss_config.topk_candicate *= 2
            
            # Adjust metrics for high resolution
            metric_iou_thresholds = [0.1, 0.25, 0.5, 0.75]  # More lenient IoU thresholds
            metric_max_detection_thresholds = [5, 50, 300]   # More detection candidates

        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.warmup_config = warmup_config
        self.num_classes = model_config.num_classes
        self.model = YOWO(model_config)

        # Modified backbone freezing for high resolution
        if freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            if self.high_resolution:
                # Keep some layers trainable for high resolution
                trainable_layers = ['layer4', 'layer3']
                for name, param in self.model.backbone_2d.named_parameters():
                    param.requires_grad = any(layer in name for layer in trainable_layers)
            else:
                for m in self.model.backbone_2d.parameters():
                    m.requires_grad = False

        if freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            if self.high_resolution:
                # Keep some layers trainable for high resolution
                trainable_layers = ['layer4', 'layer3']
                for name, param in self.model.backbone_3d.named_parameters():
                    param.requires_grad = any(layer in name for layer in trainable_layers)
            else:
                for m in self.model.backbone_3d.parameters():
                    m.requires_grad = False

        # Modified criterion for high resolution
        self.criterion = build_criterion(
            img_size=model_config.img_size,
            num_classes=model_config.num_classes,
            multi_hot=model_config.multi_hot,
            loss_cls_weight=loss_config.loss_cls_weight,
            loss_reg_weight=loss_config.loss_reg_weight * (1.5 if self.high_resolution else 1.0),
            loss_conf_weight=loss_config.loss_conf_weight,
            focal_loss=loss_config.focal_loss,
            center_sampling_radius=loss_config.center_sampling_radius,
            topk_candicate=loss_config.topk_candicate
        )

        # Modified metrics for high resolution
        self.val_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=metric_iou_thresholds,
            rec_thresholds=metric_rec_thresholds,
            max_detection_thresholds=metric_max_detection_thresholds,
            average="macro"
        )
        self.test_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=metric_iou_thresholds,
            rec_thresholds=metric_rec_thresholds,
            max_detection_thresholds=metric_max_detection_thresholds,
            average="macro"
        )

    def forward(self, video_clip: torch.Tensor):
        x = self.model(video_clip)
        return x

    def post_processing(self, outputs):
        return self.model.post_processing(outputs)

    def inference(self, video_clips: torch.Tensor) -> list[torch.Tensor]:
        return self.post_processing(
            self.forward(video_clips))

    def training_step(self, batch, batch_idx):
        frame_ids, video_clips, targets = batch
        
        # Handle high resolution inputs
        if self.high_resolution:
            # Gradient accumulation for high resolution
            self.automatic_optimization = False
            opt = self.optimizers()
            
            # Split batch for high resolution if needed
            batch_splits = 2 if video_clips.shape[0] > 2 else 1
            split_size = video_clips.shape[0] // batch_splits
            
            total_loss = 0
            opt.zero_grad()
            
            for i in range(batch_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size
                
                split_clips = video_clips[start_idx:end_idx]
                split_targets = targets[start_idx:end_idx]
                
                outputs = self.forward(split_clips)
                loss_dict = self.criterion(outputs, split_targets)
                loss = loss_dict['losses'] / batch_splits
                
                self.manual_backward(loss)
                total_loss += loss.item()
            
            opt.step()
            
            loss_dict['losses'] = total_loss
        else:
            # Original processing for standard resolution
            outputs = self.forward(video_clips)
            loss_dict = self.criterion(outputs, targets)
            total_loss = loss_dict['losses']

        # Rest of the training step remains the same
        batch_size = video_clips.size(0)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']

        out_log = {
            "lr": lr,
            "total_loss": loss_dict['losses'],
            "loss_conf": loss_dict["loss_conf"],
            "loss_cls": loss_dict["loss_cls"],
            "loss_box": loss_dict["loss_box"]
        }

        self.log_dict(
            dictionary=out_log,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True,
            batch_size=batch_size
        )
        
        return loss_dict['losses']

    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        self.eval_step(batch, mode="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        self.eval_step(batch, mode="test")

    def eval_step(self, batch, mode: Literal["val", "test"]):
        batch_img_name, batch_video_clip, batch_target = batch
        outputs = self.inference(batch_video_clip)

        # process batch gt
        gts = list(map(
            lambda x: {
                "boxes": rescale_bboxes_tensor(
                    bboxes=x["boxes"],
                    dest_width=x["orig_size"][0],
                    dest_height=x["orig_size"][1]
                ),
                "labels": x["labels"].long(),
            },
            batch_target
        ))

        # process batch predict
        preds = []
        for idx, output in enumerate(outputs):
            pred = {
                "boxes": rescale_bboxes_tensor(
                    bboxes=output[:, :4],
                    dest_width=batch_target[idx]["orig_size"][0],
                    dest_height=batch_target[idx]["orig_size"][1]
                ),
                "scores": output[:, 4],
                # int64
                "labels": output[:, 5:].long() if self.multihot else output[:, 5].long(),
            }
            preds.append(pred)

        if mode == "val":
            self.val_metric.update(preds, gts)
        else:
            self.test_metric.update(preds, gts)

    def eval_epoch(self, mode: Literal["val", "test"]):
        if mode == "val":
            result = self.val_metric.compute()
        else:
            result = self.test_metric.compute()

        metrics = {
            k: v for k, v in result.items() if k in self.include_metric_res
        }

        metrics = {
            k: v.to(self._device)
            for k, v in metrics.items() if isinstance(v, torch.Tensor)
        }

        self.log_dict(
            dictionary=metrics,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=self.trainer.num_devices > 1
        )

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch("val")
        self.val_metric.reset()

    def on_test_epoch_end(self) -> None:
        self.eval_epoch("test")
        self.test_metric.reset()

    def build_scheduler(self, config: LRSChedulerConfig, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        config_dict = asdict(config)
        config_dict['scheduler'] = config.scheduler(optimizer)
        return config_dict

    def configure_optimizers(self):
        # Modified optimizer configuration for high resolution
        if self.high_resolution:
            # Increase initial learning rate for high resolution
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.5
                
            # Modify scheduler parameters for high resolution
            self.scheduler_config.patience *= 2
            if self.warmup_config:
                self.warmup_config.num_training_steps *= 2

        return super().configure_optimizers()


class YOWOv2PP(YOWOv2Lightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot = kwargs.get("model_config", False).multi_hot

    def forward(self, video_clip, conf_threshold):
        x = super().post_processing(super().forward(video_clip))
        outputs = []
        for output in x:
            if not self.multihot:
                mask = output[..., 4] > conf_threshold
                filtered_indices = mask.nonzero(as_tuple=True)
                result = output[filtered_indices]

            else:
                # Extract confidence scores and class probabilities
                confidence_scores = output[..., 4]  # Shape (B, N)
                class_probabilities = output[..., 5:]  # Shape (B, N, 80)

                # Calculate class scores
                class_scores = torch.sqrt(
                    confidence_scores.unsqueeze(-1) * class_probabilities)  # Shape (N, 80)
                # Create a mask for scores above the threshold
                mask = class_scores > conf_threshold  # Shape (N, 80)
                keep_instance = mask.any(dim=-1)  # Shape (N,)
                bboxes = output[keep_instance, :4]
                cls_scores = class_scores[keep_instance, :]
                result = torch.cat([bboxes, cls_scores], dim=-1)
            outputs.append(result)

        return outputs
