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


def read_classnames(classname_path):
    with open(classname_path, "r") as f:
        classnames = f.read().splitlines()
        classnames = [name.strip() for name in classnames]
    return classnames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="Input source (camera id or video file path)")
    parser.add_argument("--conf", default=0.6, type=float,
                        help="Confidence threshold")
    parser.add_argument("--len-clip", default=16, type=int,
                        help="Length of frame sequence for model")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Checkpoint path")
    parser.add_argument("--cuda", action="store_true",
                        default=False, help="Use cuda")
    parser.add_argument("--classname", type=str,
                        required=True, help="Class name")
    args = parser.parse_args()

    model = YOWOv2Lightning.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        map_location=torch.device(
            "cuda") if args.cuda else torch.device("cpu")
    )

    model.eval()

    CONF_THRESH = args.conf
    CLASSNAMES = read_classnames(args.classname)
    IS_MULTIHOT = model.model.multi_hot

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
        if not IS_MULTIHOT:
            batch_results = model(inps, infer_mode=True)
            indices = batch_results[0][:, 4] > CONF_THRESH
            best_result = batch_results[0][indices]
            bboxes = best_result[:, :4]
            scores = best_result[:, 4].detach().cpu().numpy()
            labels = best_result[:, 5:].long().detach().cpu().numpy()
            bboxes = rescale_bboxes_tensor(
                bboxes,
                dest_height=height,
                dest_width=width
            ).numpy().astype(np.uint32)
            if len(scores) > 0:
                for i in range(len(scores)):
                    cv2.rectangle(vis_frame, ((bboxes[i][0]), bboxes[i][1]),
                                  (bboxes[i][2], bboxes[i][3]), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"{CLASSNAMES[int(labels[i])]}: {scores[i]:.2f}", (
                        bboxes[i][0], bboxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            act_pose = False
            batch_results = model(inps, infer_mode=True)

            cls_scores = torch.sqrt(
                batch_results[0][:, 5:] * batch_results[0][:, 4].unsqueeze(-1)
            )  # class confidence * detection confidence

            keep_instance = torch.any(cls_scores > CONF_THRESH, dim=-1)
            bboxes = rescale_bboxes_tensor(
                bboxes=batch_results[0][keep_instance, :4],
                dest_height=height,
                dest_width=width
            )
            cls_scores = cls_scores[keep_instance, :]

            for bbox, cls_score in zip(bboxes, cls_scores):
                indices = torch.where(cls_score > CONF_THRESH)[0].cpu().numpy()
                x1, y1, x2, y2 = bbox.cpu().numpy().astype(np.uint32)
                # draw bbox
                cv2.rectangle(vis_frame, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)
                # draw text
                blk = np.zeros(vis_frame.shape, np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text = []
                text_size = []
                for _, cls_ind in enumerate(indices):
                    text.append("[{:.2f}] ".format(
                        cls_score[cls_ind]) + str(CLASSNAMES[cls_ind]))
                    text_size.append(cv2.getTextSize(
                        text[-1], font, fontScale=0.5, thickness=1)[0])
                    coord.append((x1+3, y1+14+20*_))
                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-12), (coord[-1][0]+text_size[-1]
                                                                           [0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                vis_frame = cv2.addWeighted(vis_frame, 1.0, blk, 0.5, 1)
                for t in range(len(text)):
                    cv2.putText(
                        vis_frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)

        cv2.imshow("frame", vis_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
