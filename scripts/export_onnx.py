import argparse
import torch
from yowo.models import YOWOv2PP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="path to checkpoint")
    parser.add_argument("--len-clip", type=int, default=16,
                        help="length of input clip")
    parser.add_argument("--onnx", type=str, required=True,
                        help="path to onnx file")
    args = parser.parse_args()

    plmodel = YOWOv2PP.load_from_checkpoint(args.ckpt)
    is_multihot = plmodel.model.multi_hot
    dummy_input = (torch.randn(1, 3, args.len_clip, 224, 224), 0.5)

    output_names = ["outputs"]

    # dynamic_batch = {
    #     "input": [0],
    #     "outputs": [0]
    # }

    plmodel.to_onnx(
        file_path=args.onnx,
        input_sample=dummy_input,
        input_names=['input', 'conf_threshold'],
        output_names=output_names,
        export_params=True,
        # dynamic_axes=dynamic_batch
    )
