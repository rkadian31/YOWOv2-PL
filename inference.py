from collections import deque
import torch
import cv2
import numpy as np
from yowo.models import YOWOv2Lightning
from yowo.utils.box_ops import rescale_bboxes_tensor
import argparse


def preprocess_input(imgs: list[np.ndarray]):
    inps = [cv2.resize(img, (224, 224)) for img in imgs]
    return torch.tensor(np.stack(inps)).permute(3, 0, 1, 2).unsqueeze(0).float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="Input source (camera id or video file path)")
    parser.add_argument("--conf", default=0.7, type=float,
                        help="Confidence threshold")
    parser.add_argument("--len-clip", default=16, type=int,
                        help="Length of frame sequence for model")
    parser.add_argument("--checkpoint", default="checkpoints/yowo_v2_tiny_ucf24_pl_cvt.ckpt",
                        help="Checkpoint path")
    parser.add_argument("--cuda", action="store_true",
                        default=False, help="Use cuda")
    args = parser.parse_args()

    CONF_THRESH = args.conf
    model = YOWOv2Lightning.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        map_location=torch.device(
            "cuda") if args.cuda else torch.device("cpu")
    )

    model.eval()
    frames = deque(maxlen=args.len_clip)
    cap = cv2.VideoCapture(args.source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        # print(height, width) # 240, 320

        if len(frames) <= 0:
            for _ in range(args.len_clip):
                frames.append(frame)

        frames.append(frame)
        frames.popleft()

        inps = preprocess_input(frames)
        vis_frame = frames[-1].copy()
        # print(inp.shape)
        batch_scores, batch_labels, batch_bboxes = model(inps, infer_mode=True)

        best_cand = batch_scores[0] > CONF_THRESH
        if torch.any(best_cand):
            bboxes = rescale_bboxes_tensor(
                batch_bboxes[0][best_cand, ...],
                dest_height=height,
                dest_width=width
            ).numpy().astype(np.uint32)
            scores = batch_scores[0][best_cand].numpy()
            labels = batch_labels[0][best_cand].numpy() + 1

            for i in range(len(scores)):
                cv2.rectangle(vis_frame, ((bboxes[i][0]), bboxes[i][1]),
                              (bboxes[i][2], bboxes[i][3]), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"{labels[i]}: {scores[i]:.2f}", (
                    bboxes[i][0], bboxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("frame", vis_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
