import argparse
import torch
from yowo.models import YOWOv2Lightning


class ONNXYOWOv2(YOWOv2Lightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot = kwargs.get("model_config", False).multi_hot

    def forward(self, video_clip, conf_threshold):
        x = super().post_processing(super().forward(video_clip))
        x = torch.stack(x)
        if not self.multihot:
            mask = x[..., 4] > conf_threshold
            filtered_indices = mask.nonzero(as_tuple=True)
            outputs = x[filtered_indices]

        else:
            # Extract confidence scores and class probabilities
            confidence_scores = x[0][..., 4]  # Shape (B, N)
            class_probabilities = x[0][..., 5:]  # Shape (B, N, 80)

            # Calculate class scores
            class_scores = torch.sqrt(
                confidence_scores.unsqueeze(-1) * class_probabilities)  # Shape (N, 80)
            # Create a mask for scores above the threshold
            mask = class_scores > conf_threshold  # Shape (N, 80)
            keep_instance = mask.any(dim=-1)  # Shape (N,)
            bboxes = x[0][keep_instance, :4]
            cls_scores = class_scores[keep_instance, :]
            outputs = torch.cat([bboxes, cls_scores], dim=-1)

        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to checkpoint")
    parser.add_argument("--len-clip", type=int, default=16,
                        help="length of input clip")
    parser.add_argument("--onnx", type=str, required=True,
                        help="path to onnx file")
    args = parser.parse_args()

    plmodel = ONNXYOWOv2.load_from_checkpoint(args.ckpt)
    is_multihot = plmodel.model.multi_hot
    dummy_input = (torch.randn(1, 3, args.len_clip, 224, 224), 0.5)

    output_names = ["outputs"]

    # dynamic_axes = {
    #     'input': {0: 'batch_size'},    # variable length axes
    #     'bboxes': {0: 'batch_size'},
    #     'scores': {0: 'batch_size'},
    #     'labels': {0: 'batch_size'}
    # }

    plmodel.to_onnx(
        file_path=args.onnx,
        input_sample=dummy_input,
        input_names=['input', 'conf_threshold'],
        output_names=output_names,
        export_params=True,
        # dynamic_axes=dynamic_axes
    )
