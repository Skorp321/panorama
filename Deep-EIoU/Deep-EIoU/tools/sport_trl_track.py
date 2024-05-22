import argparse
import os
import sys

import torch
from utils import Timer
from utils import TeamClassifier, HomographySetup, DatabaseWriter
from utils import write_results, get_crops, image_track, apply_homography_to_point

import cv2
from tqdm.auto import tqdm

sys.path.append(".")
import numpy as np
from loguru import logger
from tracker.tracking_utils.timer import Timer
from tracker.Deep_EIoU import Deep_EIoU
from ultralytics import YOLO
import ffmpegcv
from torchreid.utils import FeatureExtractor
from sklearn.cluster import KMeans
import pandas as pd

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU For Evaluation!")

    parser.add_argument(
        "--path",
        default="/home/skorp321/Projects/panorama/data/Swiss_vs_Slovakia-panoramic_video.mp4",
        help="path to images or video",
    )
    parser.add_argument(
        "--show",
        default=False,
        help="Show the processed images",
    )
    parser.add_argument(
        "--output_db",
        default="/home/skorp321/Projects/panorama/data/soccer_analitics.db",
        help="path to bd",
    )
    parser.add_argument(
        "--path_to_field",
        default="/home/skorp321/Projects/panorama/data/soccer_field.png",
        help="path to soccer field image",
    )
    parser.add_argument(
        "--path_to_field_points",
        default="/home/skorp321/Projects/panorama/data/soccer_field.png",
        help="path to soccer field image",
    )
    parser.add_argument(
        "--h_matrix_path",
        default="",#/home/skorp321/Projects/panorama/data/h_matrix_path.npy",
        help="path to soccer field image",
    )
    parser.add_argument(
        "--path_to_det",
        default="/home/skorp321/Projects/panorama/models/yolov8m_goalkeeper_1280.pt",
        help="path to detector model",
    )
    parser.add_argument(
        "--path_to_keypoints_det",
        default="/home/skorp321/Projects/panorama/runs/pose/train8/weights/best.pt",
        help="path to detector model",
    )
    parser.add_argument(
        "--ball_det",
        default="/home/skorp321/Projects/panorama/models/ball_SN5+52games.pt",
        help="path to detector model",
    )
    parser.add_argument(
        "--path_to_reid",
        default="/home/skorp321/Projects/panorama/models/osnet_ain_x1_0_triplet_custom.pt",
        help="path to reid model",
    )
    parser.add_argument(
        "--save_path", default="/home/skorp321/Projects/panorama/data/output", type=str
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--save-frames",
        dest="save_frames",
        default=False,
        action="store_true",
        help="save sequences with tracks.",
    )
    # Homography

    # Detector
    parser.add_argument(
        "--device",
        default="0",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=1280, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )

    # tracking args
    parser.add_argument(
        "--track_high_thresh",
        type=float,
        default=0.6,
        help="tracking confidence threshold",
    )
    parser.add_argument(
        "--track_low_thresh",
        default=0.1,
        type=float,
        help="lowest detection threshold valid for tracks",
    )
    parser.add_argument(
        "--new_track_thresh", default=0.7, type=float, help="new track thresh"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=60, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.8,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--min_box_area", type=float, default=10, help="filter out tiny boxes"
    )
    parser.add_argument("--nms_thres", type=float, default=0.7, help="nms threshold")
    # ReID
    parser.add_argument(
        "--with-reid",
        dest="with_reid",
        default=True,
        action="store_true",
        help="use Re-ID flag.",
    )
    parser.add_argument(
        "--fast-reid-config",
        dest="fast_reid_config",
        type=str,
        help="reid config file path",
    )
    parser.add_argument(
        "--fast-reid-weights",
        dest="fast_reid_weights",
        default=r"pretrained/mot17_sbs_S50.pth",
        type=str,
        help="reid config file path",
    )
    parser.add_argument(
        "--proximity_thresh",
        type=float,
        default=0.5,
        help="threshold for rejecting low overlap reid matches",
    )
    parser.add_argument(
        "--appearance_thresh",
        type=float,
        default=0.25,
        help="threshold for rejecting low appearance similarity reid matches",
    )

    return parser


def main():

    timer = Timer()
    collors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (125, 125, 0)]
    args = make_parser().parse_args()

    # Detector
    model = YOLO(args.path_to_det)
    print(model.names)
    ball_model = YOLO(args.ball_det)

    # ReID
    model_reid = TeamClassifier(weights_path=args.path_to_reid, model_name="osnet_x1_0")

    # Tracker
    tracker = Deep_EIoU(args, frame_rate=60)

    # Homography
    homographer = HomographySetup(args)
    H = homographer.compute_homography_matrix()

    #Database connector and writer
    bd = DatabaseWriter(args)

    cap = ffmpegcv.VideoCaptureNV(args.path)
    fps = cap.fps

    save_path = os.path.join(args.save_path, args.path.split("/")[-1])

    vid_writer = ffmpegcv.VideoWriterNV(save_path, "h264", fps)
    logger.info(f"save path to video: {save_path}")

    text_scale = 2
    size = cap.size
    count = 1

    for frame in tqdm(cap):

        timer.tic()

        img_copy = frame.copy()
        img_layout_copy = homographer.layout_img.copy()

        if args.fp16:
            img_copy = img_copy.half()  # to FP16

        if count % 1 == 0:
            dets = []
            imgs_list = []
            cls_list = []
            embed = {"cls": [], "embs": []}
            
            logger.info(f"Processing seq {count} in {size}")

            imgs, offsets = get_crops(img_copy)
            outputs = model(
                imgs,
                imgsz=1280,
                show_conf=False,
                show_boxes=False,
                device=0,
                stream=False,
                agnostic_nms = True,
                max_det = 26
            )
            '''ball_output = ball_model(
                imgs,
                imgsz=1280,
                show_conf=False,
                show_boxes=False,
                device=0,
                stream=False,
            )'''
            
            #bal_box = ball_output[0].boxes.data[0]

            # Process results list
            for offset, result in zip(offsets, outputs):#, ball_output):
                boxes = result.boxes  # Boxes object for bounding box outputs
                '''ball_box = ball.boxes.data.detach().cpu().tolist()
                print(ball_box)      
                if len(ball_box) > 0:
                    ball_box = ball_box[0]
                    bx1, by1 = int(offset + ball_box[0]), int(ball_box[1])
                    bx2, by2 = int(offset + ball_box[2]), int(ball_box[3])
                    cv2.rectangle(
                        img_copy, (int(bx1), int(by1)), (int(bx2), int(by2)), (255,255,255), 1
                    )'''
                for conf, box in zip(boxes.conf, boxes.data):
                    x1, y1 = int(offset + box[0]), int(box[1])
                    x2, y2 = int(offset + box[2]), int(box[3])

                        
                    conf = conf.detach().cpu().tolist()
                    cls = int(box[5])
                    collor = collors[cls]
                    cls_list.append(cls)
                    imgs_list.append(img_copy[y1:y2, x1:x2])

                    dets.append([count, cls, x1, y1, x2, y2, conf])

            columns = ["frame", "cls", "x1", "y1", "x2", "y2", "conf"]

            frame_data = pd.DataFrame(dets, columns=columns)

            embed["cls"] = cls_list
            embed["embs"] = model_reid.extract_features(imgs_list)
            embeddings = pd.DataFrame(embed)

            embeddings["team"] = model_reid.classify(
                embeddings["embs"].to_list(), count
            )
            frame_data["team"] = embeddings["team"]
            frame_data.loc[frame_data["cls"] == 2, "team"] = 2
            arr = np.array(dets, dtype=np.float64)
            team1, team2 = 0, 0
            h_point_x = []
            h_point_y = []
            for i, (cls, emb, team) in embeddings.iterrows():

                box = arr[i, 2:-1]
                conf = arr[i, -1]
                x1, y1, x2, y2 = box
                xm = int(x1 + ((x2 - x1) / 2.0))

                if cls != 2:
                    if team == 0:
                        collor = collors[0]
                        team1 += 1
                    if team == 1:
                        collor = collors[1]
                        team2 += 1

                    cv2.rectangle(
                        img_copy, (int(x1), int(y1)), (int(x2), int(y2)), collor, 1
                    )

                    h_point = apply_homography_to_point([xm, y2], H)
                    h_point_x.append(h_point[0])
                    h_point_y.append(h_point[1])
                    cv2.circle(img_layout_copy, h_point, 6, collor, -1)

                else:
                    h_point = apply_homography_to_point([xm, y2], H)
                    h_point_x.append(h_point[0])
                    h_point_y.append(h_point[1])
                    cv2.circle(img_layout_copy, h_point, 6, collors[2], -1)
                    cv2.rectangle(
                        img_copy, (int(x1), int(y1)), (int(x2), int(y2)), collors[2], 1
                    )

            frame_data["xm"], frame_data["ym"] = h_point_x, h_point_y
            frame_data.sort_values(by=["x1", "y1"], inplace=True)
            #print(f"Team 1: {team1}, team 2: {team2}")
            detections = arr[:, 2:]
            
            results = image_track(
                tracker, detections, embeddings["embs"], args.save_path, args, count
            )
            track_columns = [
                "frame_id",
                "id",
                "x1",
                "y1",
                "w",
                "h",
                "conf",
                "none1",
                "none2",
                "none3",
            ]
            data = [tuple(res.split(',')) for res in results]
            results_df = pd.DataFrame(data, columns=track_columns)
            results_df = results_df.loc[:, ["id", "x1", "y1"]]
            results_df['x1'] = results_df['x1'].astype(float)    
            results_df['id'] = results_df['id'].astype(int)   
            results_df = results_df.sort_values(by=["x1"])
            frame_data = pd.concat([frame_data, results_df], axis=1)

            timer.toc()

            for num_res in results:

                res = num_res.split(",")
                cv2.putText(
                    img_copy,
                    res[1],
                    (int(float(res[2])), int(float(res[3]))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                img_copy,
                "frame: %d fps: %.2f num: %d"
                % (count, 1.0 / timer.average_time, detections.shape[0]),
                (0, int(15 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                thickness=2,
            )

            _, _, concatenated_img = homographer.prepare_images_for_display(
                img_copy, img_layout_copy
            )

            if args.show:
                cv2.imshow('Video', concatenated_img)

                # Добавляем задержку для показа видео в реальном времени
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            count += 1
            vid_writer.write(concatenated_img)
            bd.update_db(frame_data)
        else:
            timer.toc()
            cv2.putText(
                img_copy,
                "frame: %d fps: %.2f num: %d"
                % (count, 1.0 / timer.average_time, detections.shape[0]),
                (0, int(15 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                thickness=2,
            )
            _, _, concatenated_img = homographer.prepare_images_for_display(img_copy)
            
            if args.show:
                cv2.imshow("Video", concatenated_img)
                # Добавляем задержку для показа видео в реальном времени
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            count += 1
            vid_writer.write(concatenated_img)
        
    bd.close_db()
    cap.release()
    vid_writer.release()


if __name__ == "__main__":
    main()
