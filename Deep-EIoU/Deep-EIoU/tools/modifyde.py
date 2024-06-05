import argparse
import os
import sys

from copy import deepcopy
from utils import (
    TeamClassifier, 
    HomographySetup, 
    DatabaseWriter, 
    Timer, 
    TrackMacher)
from utils import (
    write_results, 
    get_crops, 
    image_track, 
    apply_homography_to_point, 
    make_parser)

import cv2
from tqdm.auto import tqdm

sys.path.append(".")
import numpy as np
from loguru import logger
from tracker.tracking_utils.timer import Timer
from tracker.Deep_EIoU import Deep_EIoU
from ultralytics import YOLO
import ffmpegcv
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
    #ball_model = YOLO(args.ball_det)

    # ReID
    model_reid = TeamClassifier(weights_path=args.path_to_reid, model_name="osnet_x1_0")

    # Tracker
    tracker = Deep_EIoU(args, frame_rate=60)

    # Homography
    homographer = HomographySetup(args)
    H = homographer.compute_homography_matrix()

    #Database connector and writer
    bd = DatabaseWriter(args)
    
    macher = TrackMacher()

    save_path = os.path.join(args.save_path, args.path.split("/")[-1])
    
    cap = ffmpegcv.VideoCaptureNV(args.path)
    #cap = cv2.VideoCapture(args.path)
    fps = cap.fps
    vid_writer = ffmpegcv.VideoWriterNV(save_path, "h264", fps)
    logger.info(f"save path to video: {save_path}")
    
    text_scale = 2
    #size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = cap.size
    count = 1
    prev_dets = pd.DataFrame()

    '''while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break'''
                
    for frame in tqdm(cap):
        
        timer.tic()

        img_copy = frame.copy()
        img_layout_copy = homographer.layout_img.copy()
        
        if args.fp16:
            img_copy = img_copy.half()  # to FP16
            
        if count <= args.init_tresh:
            dets = []
            imgs_list = []
            cls_list = []
            embed = {"cls": [], "embs": []}
            
            imgs, offsets = get_crops(img_copy, offset=30)
            outputs = model(imgs,
                            imgsz=1280,
                            show_conf=False,
                            show_boxes=False,
                            device=0,
                            stream=False,
                            agnostic_nms = True,
                            max_det = 26
                        )
            
            for offset, result in zip(offsets, outputs):#, ball_output):
                boxes = result.boxes  # Boxes object for bounding box outputs
                
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

            dets = pd.DataFrame(dets, columns=columns)

            embed["cls"] = cls_list
            embed["embs"] = model_reid.extract_features(imgs_list)
            embeddings = pd.DataFrame(embed)
            embeddings["team"] = model_reid.classify(
                embeddings["embs"].to_list(), count
            )
            detections = dets.to_numpy()
            results = image_track(
                tracker, detections[:,1:], embeddings["embs"], args, count
            )
            
            track_columns = [
                "frame_id",
                "id",
                "x1",
                "y1",
                "w",
                "h",
                "conf",
                "cls",
                "none2",
                "none3",
            ]
            data = [tuple(res.split(',')) for res in results]
            results_df = pd.DataFrame(data, columns=track_columns)
            embeddings['id'] = results_df['id']
            count += 1
            prev_dets = deepcopy(results_df)
        else:
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
                
                for offset, result in zip(offsets, outputs):#, ball_output):
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    
                    for conf, box in zip(boxes.conf, boxes.data):
                        x1, y1 = int(offset + box[0]), int(box[1])
                        x2, y2 = int(offset + box[2]), int(box[3])
                            
                        conf = conf.detach().cpu().tolist()
                        cls = int(box[5])
                        collor = collors[cls]
                        cls_list.append(cls)
                        imgs_list.append(img_copy[y1:y2, x1:x2])

                        dets.append([count, cls, x1, y1, x2, y2, conf])
                        
                embed["cls"] = cls_list
                embed["embs"] = model_reid.extract_features(imgs_list)
                embeddings = pd.DataFrame(embed)           
                
                dets = pd.DataFrame(dets, columns=["frame", "cls", "x1", "y1", "x2", "y2", "conf"])
                detections = dets.to_numpy()
                results = image_track(tracker, detections[:,1:7], embeddings["embs"], args, count)
                
                track_columns = [
                    "frame_id",
                    "id",
                    "x1",
                    "y1",
                    "w",
                    "h",
                    "conf",
                    "cls",
                    "x_anchor",
                    "y_anchor",
                ]
                data = [tuple(res.split(',')) for res in results]
                results_df = pd.DataFrame(data, columns=track_columns)
                results_df['x1'] = results_df['x1'].astype(float).astype(int)
                results_df['y1'] = results_df['y1'].astype(float).astype(int)
                dets['x1'] = dets['x1'].astype(int)
                dets['y1'] = dets['y1'].astype(int)

                df = pd.merge(results_df[['id', 'x1', 'y1', 'w', 'h']], dets[['frame', 'x1', 'y1', 'conf', 'cls']], on=['x1', 'y1'], how='inner')
                df['h'] = df['h'].astype(float).astype(int)
                df['w'] = df['w'].astype(float).astype(int)
                df['id'] = df['id'].astype(int)
                df['x_anchor'] = (df['x1'] + (df['w'] / 2.0)).astype(int)
                df['y_anchor'] = (df['y1'] + df['h']).astype(int)

                macht_df = macher.mach(pd.DataFrame(df.head(23)), prev_dets)
                timer.toc()
                for i, row in macht_df.iterrows():

                    x1, y1 = int(row['x1']), int(row['y1'])
                    x2, y2 = int(x1 + row['w']), int(y1 +row['h'])
                    xm = int(row['x_anchor'])
                    ym = int(row['y_anchor'])
                    h_point = apply_homography_to_point([xm, ym], H)
                    cv2.circle(img_layout_copy, h_point, 6, collors[2], -1)
                    cv2.rectangle(
                        img_copy, (int(x1), int(y1)), (int(x2), int(y2)), collors[2], 1
                    )
                    cv2.putText(img_copy, str(int(row['id'])), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(img_layout_copy, str(int(row['id'])), h_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(
                        img_copy,
                        "frame: %d fps: %.2f num: %d"
                        % (count, 1.0 / timer.average_time, macht_df.shape[0]),
                        (0, int(15 * text_scale)),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255),
                        thickness=2,
                    )
                    _, _, concatenated_img = homographer.prepare_images_for_display(img_copy, img_layout_copy)

                if args.show:
                    #cv2.imshow('field', img_copy)
                    #cv2.imshow('layout', img_layout_copy)
                    cv2.imshow('Video', concatenated_img)

                    # Добавляем задержку для показа видео в реальном времени
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
                vid_writer.write(concatenated_img)
                prev_dets = deepcopy(macht_df)
                count += 1
    cap.release()
    vid_writer.release()

if __name__ == "__main__":
    main()