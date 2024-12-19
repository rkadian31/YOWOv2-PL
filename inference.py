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
    parser.add_argument("--conf", default=0.7, type=float,
                        help="Confidence threshold")
    parser.add_argument("--len-clip", default=16, type=int,
                        help="Length of frame sequence for model")
    parser.add_argument("--checkpoint", default="checkpoints/yowo_v2_tiny_ucf24_pl_cvt.ckpt",
                        help="Checkpoint path")
    parser.add_argument("--cuda", action="store_true",
                        default=False, help="Use cuda")
    parser.add_argument("--multihot", action="store_true",
                        default=False, help="Use multi-hot")
    parser.add_argument("--classname", type=str,
                        required=True, help="Class name")
    args = parser.parse_args()

    CONF_THRESH = args.conf
    model = YOWOv2Lightning.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        map_location=torch.device(
            "cuda") if args.cuda else torch.device("cpu")
    )

    CLASSNAMES = read_classnames(args.classname)

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
        if not args.multihot:
            batch_scores, batch_labels, batch_bboxes = model(
                inps, infer_mode=True)

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
                    cv2.putText(vis_frame, f"{CLASSNAMES[labels[i]]}: {scores[i]:.2f}", (
                        bboxes[i][0], bboxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            act_pose = False
            batch_results = model(inps, infer_mode=True)
            for bbox in batch_results[0].numpy():
                x1, y1, x2, y2 = bbox[:4]
                if act_pose:
                    # only show 14 poses of AVA.
                    cls_conf = bbox[5:5+14]
                else:
                    # show all actions of AVA.
                    cls_conf = bbox[5:]

                # rescale bbox
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)

                # score = obj * cls
                det_conf = float(bbox[4])
                cls_scores = np.sqrt(det_conf * cls_conf)

                indices = np.where(cls_scores > CONF_THRESH)
                scores = cls_scores[indices]
                indices = list(indices[0])
                scores = list(scores)

                if len(scores) > 0:
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
                            scores[_]) + str(CLASSNAMES[cls_ind]))
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
