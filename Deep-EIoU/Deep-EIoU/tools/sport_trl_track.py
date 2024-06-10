import argparse
import os
import sys

import torch
from utils import Timer
from utils import TeamClassifier, HomographySetup, DatabaseWriter
from utils import write_results, get_crops, image_track, apply_homography_to_point, make_parser

import cv2
from tqdm.auto import tqdm

sys.path.append(".")
import numpy as np
from loguru import logger
from tracker.Deep_EIoU import Deep_EIoU
from ultralytics import YOLO
import ffmpegcv
from torchreid.utils import FeatureExtractor
from sklearn.cluster import KMeans
import pandas as pd

# Global
trackerTimer = Timer()


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
                    elif team == 1:
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
            frame_data = frame_data.sort_values(by=["x1", "y1"])
            #print(f"Team 1: {team1}, team 2: {team2}")
            detections = arr[:, 2:]

            results = image_track(
                tracker, detections, embeddings["embs"], args, count
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
            vid_writer.write(concatenated_img)

        count += 1
    bd.close_db()
    cap.release()
    vid_writer.release()


if __name__ == "__main__":
    main()
