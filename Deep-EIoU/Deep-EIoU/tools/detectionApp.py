import argparse
import os
import sys

from copy import deepcopy

import torch

from utils import (
    TeamClassifier,
    HomographySetup,
    DatabaseWriter,
    Timer,
    TrackMacher,
    teamMatcher,
)

from utils import (
    write_results,
    get_crops,
    image_track,
    apply_homography_to_point,
    make_parser,
    non_max_suppression,
    draw_annos,
    drow_metrics,
)

import cv2
from tqdm.auto import tqdm
import streamlit as st

sys.path.append(".")
import numpy as np
from loguru import logger

from tracker.Deep_EIoU import Deep_EIoU
from ultralytics import YOLO
import ffmpegcv
import torch_tensorrt

from sklearn.cluster import KMeans
import pandas as pd

from dataclasses import dataclass


# Global
trackerTimer = Timer()


@dataclass
class DataArgs:
    path: str = "/container_dir/data/Swiss_vs_Slovakia-panoramic_video.mp4"
    show: bool = True
    output_db: str = "/container_dir/data/soccer_analitics.db"
    path_to_field: str = "/container_dir/data/soccer_field.png"
    path_to_field_points: str = (
        "/container_dir/data/soccer_field_anno/annotations/person_keypoints_default.json"
    )
    path_to_layout_points: str = (
        "/container_dir/data/Swiss_vs_Slovakia-panoramic_video_anno/annotations/person_keypoints_default.json"
    )
    h_matrix_path: str = "/container_dir/data/h_matrix_path.npy"
    path_to_det: str = "/container_dir/models/yolov8m_goalkeeper_1280.engine"
    path_to_keypoints_det: str = "/container_dir/runs/pose/train8/weights/best.pt"
    ball_det: str = "/container_dir/models/ball_SN5+52games.pt"
    path_to_reid: str = "/container_dir/models/osnet_ain_x1_0_triplet_custom.pt"
    save_path: str = "/container_dir/data/output"
    output_db: str = "/container_dir/data/soccer_analitics.db"
    trt: bool = False
    save_frames: bool = False
    init_tresh: int = 3
    device: int = 0
    conf: float = 0.5
    nms: float = 0.8
    tsize: int = 1280
    fp16: bool = False
    fuse: bool = False
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 100
    match_thresh: float = 0.7
    aspect_ratio_thresh: float = 1.6
    min_box_area: float = 4
    nms_thres: float = 0.7
    with_reid: bool = False
    fast_reid_config: str = ""
    fast_reid_weights: str = r"pretrained/mot17_sbs_S50.pth"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25


def detect(cap, stframe, output_file_name, save_output, plot_hyperparser, df_field):

    timer = Timer()
    collors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    args = DataArgs()  # make_parser().parse_args()
    # teams_stats = pd.DataFrame({'id': list(range(24)), 'team1': [0]*23, 'team2': [0]*23,  'team':  [0]*23})
    if output_file_name:
        args.save_path = output_file_name + ".mp4"
    # Detector
    model = YOLO(args.path_to_det)
    # ball_model = YOLO(args.ball_det)

    # ReID
    model_reid = TeamClassifier(weights_path=args.path_to_reid, model_name="osnet_x1_0")

    if args.trt:
        path_trt = args.path_to_reid.replace(".pt", ".engine")
        trt_ts_module = torch.jit.load(path_trt)

        model_reid.extractor.model = trt_ts_module
    # Team Matcher
    team_matcher = teamMatcher()
    # Tracker
    tracker = Deep_EIoU(args, frame_rate=60)

    # Homography
    homographer = HomographySetup(args)
    H = homographer.compute_homography_matrix()

    # Database connector and writer
    bd = DatabaseWriter(args)

    macher = TrackMacher()

    save_path = os.path.join(args.save_path, args.path.split("/")[-1])

    cap = ffmpegcv.VideoCaptureNV(args.path)
    # cap = cv2.VideoCapture(args.path)
    fps = cap.fps
    vid_writer = ffmpegcv.VideoWriterNV(save_path, "h264", fps)
    logger.info(f"save path to video: {save_path}")

    text_scale = 1
    # size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_prog_bar = st.progress(0, text="Detection starting.")
    size = cap.size
    count = 1

    prev_dets = pd.DataFrame()

    """while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break"""

    for frame in tqdm(cap):

        timer.tic()
        percent_complete = int(count / (cap.count) * 100)
        img_copy = frame.copy()  # [int(size[1]*0.4):, :].copy()
        img_layout_copy = homographer.layout_img.copy()

        if args.fp16:
            img_copy = img_copy.astype("float16")  # to FP16

        if count <= args.init_tresh:
            imgs_list = []
            cls_list = []
            embed = {"cls": [], "embs": []}

            imgs, offsets = get_crops(img_copy)
            outputs = model(
                imgs,
                imgsz=args.tsize,
                show_conf=False,
                show_boxes=False,
                device=0,
                stream=False,
                agnostic_nms=True,
                max_det=26,
                half=True,
            )

            dets = []
            for offset, result in zip(offsets, outputs):  # , ball_output):
                boxes = result.boxes  # Boxes object for bounding box outputs

                for conf, box in zip(boxes.conf, boxes.data):
                    x1, y1 = int(offset + box[0]), int(box[1])
                    x2, y2 = int(offset + box[2]), int(box[3])

                    conf = conf.detach().cpu().tolist()
                    cls = int(box[5])
                    cls_list.append(cls)
                    imgs_list.append(img_copy[y1:y2, x1:x2])

                    dets.append([count, cls, x1, y1, x2, y2, conf])

            columns = ["frame", "cls", "x1", "y1", "x2", "y2", "conf"]

            dets = pd.DataFrame(dets, columns=columns)

            dets = non_max_suppression(
                dets[["cls", "x1", "y1", "x2", "y2", "conf"]], 0.5
            )

            embed["cls"] = cls_list
            embed["embs"] = model_reid.extract_features(imgs_list)
            embeddings = pd.DataFrame(embed)
            embeddings["team"] = model_reid.classify(
                embeddings["embs"].to_list(), count
            )

            results = image_track(tracker, dets, embeddings["embs"], args, count)

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
            data = [tuple(res.split(",")) for res in results]
            results_df = pd.DataFrame(data, columns=track_columns)
            embeddings["id"] = results_df["id"]
            prev_dets = deepcopy(results_df)

        elif count % 1 == 0:
            dets = []
            imgs_list = []
            cls_list = []
            embed = {"cls": [], "embs": []}

            logger.info(f"Processing seq {count} in {size}")

            imgs, offsets = get_crops(img_copy)
            outputs = model(
                imgs,
                imgsz=args.tsize,
                show_conf=False,
                show_boxes=False,
                device=0,
                stream=False,
                agnostic_nms=True,
                max_det=26,
                half=True,
            )

            for offset, result in zip(offsets, outputs):  # , ball_output):
                boxes = result.boxes  # Boxes object for bounding box outputs

                for conf, box in zip(boxes.conf, boxes.data):
                    x1, y1 = int(offset + box[0]), int(box[1])
                    x2, y2 = int(offset + box[2]), int(box[3])

                    conf = conf.detach().cpu().tolist()
                    cls = int(box[5])
                    dets.append([count, cls, x1, y1, x2, y2, conf])

            dets = pd.DataFrame(
                dets, columns=["frame", "cls", "x1", "y1", "x2", "y2", "conf"]
            )
            dets_nms = non_max_suppression(
                dets[["cls", "x1", "y1", "x2", "y2", "conf"]], 0.4
            )
            dets_nms = dets_nms[:23]
            imgs_list = [
                img_copy[int(y1) : int(y2), int(x1) : int(x2)]
                for (x1, y1, x2, y2, _, _) in dets_nms
            ]
            cls_list = [int(cls) for (_, _, _, _, _, cls) in dets_nms]
            embed["cls"] = cls_list

            embed["embs"] = model_reid.extract_features(imgs_list)
            embeddings = pd.DataFrame(embed)
            results = image_track(tracker, dets_nms, embeddings["embs"], args, count)

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
            dets_nms = pd.DataFrame(
                dets_nms,
                columns=["x1", "y1", "x2", "y2", "conf", "cls"],
            )
            dets_nms["frame"] = np.array([count] * dets_nms.shape[0])

            embeddings["x1"] = dets_nms["x1"].astype(float).astype(int)
            embeddings["y1"] = dets_nms["y1"].astype(float).astype(int)
            embeddings["w"] = dets_nms["x2"].astype(float).astype(int) - dets_nms[
                "x1"
            ].astype(float).astype(int)

            data = [tuple(res.split(",")) for res in results]
            results_df = pd.DataFrame(
                data,
                columns=track_columns,
            )
            results_df["id"] = results_df["id"].astype(float).astype(int)
            results_df["x1"] = results_df["x1"].astype(float).astype(int)
            results_df["y1"] = results_df["y1"].astype(float).astype(int)
            results_df["w"] = results_df["w"].astype(float).astype(int)

            # embeddings['team'] = team_matcher.update_matches(results_df['id'].to_numpy(),
            #                                                 teams_labels)

            results_df = pd.merge(
                results_df[["id", "x1", "y1", "w", "h"]],
                embeddings[["x1", "y1", "w", "embs"]],
                on=["x1", "y1", "w"],
                how="inner",
            )
            results_df.sort_values(["id"], axis=0, inplace=True)
            results_df.drop_duplicates(subset="id", inplace=True)
            teams_labels = model_reid.classify(results_df["embs"].to_list(), count)

            results_df["team"] = team_matcher.update_matches(
                results_df["id"].to_numpy(), teams_labels[:23]
            )

            dets_nms["x1"] = dets_nms["x1"].astype(int)
            dets_nms["y1"] = dets_nms["y1"].astype(int)
            dets_nms["w"] = dets_nms["x2"].astype(int) - dets_nms["x1"].astype(int)
            dets_nms["cls"] = dets_nms["cls"].astype(int)

            df = pd.merge(
                results_df[["id", "x1", "y1", "w", "h", "team"]],
                dets_nms[["frame", "x1", "y1", "w", "conf", "cls"]],
                on=["x1", "y1", "w"],
                how="inner",
            )

            df["h"] = df["h"].astype(float).astype(int)
            df["w"] = df["w"].astype(float).astype(int)
            df["id"] = df["id"].astype(int)
            df["x_anchor"] = (df["x1"] + (df["w"] / 2.0)).astype(int)
            df["y_anchor"] = (df["y1"] + df["h"]).astype(int)

            macht_df = macher.mach(pd.DataFrame(df.head(23)), prev_dets)
            timer.toc()
            # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            for i, row in macht_df.iterrows():
                x1, y1 = int(row["x1"]), int(row["y1"])
                x2, y2 = int(x1 + row["w"]), int(y1 + row["h"])
                xm = int(row["x_anchor"])
                ym = int(row["y_anchor"])
                h_point = apply_homography_to_point([xm, ym], H)

                if row["cls"] != 2:
                    draw_annos(
                        collors[int(row["team"])],
                        img_copy,
                        img_layout_copy,
                        x1,
                        y1,
                        x2,
                        y2,
                        row,
                        h_point,
                    )
                else:
                    draw_annos(
                        collors[3],
                        img_copy,
                        img_layout_copy,
                        x1,
                        y1,
                        x2,
                        y2,
                        row,
                        h_point,
                    )

            # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            _, _, concatenated_img = homographer.prepare_images_for_display(
                img_copy, img_layout_copy
            )

            concatenated_img = drow_metrics(
                concatenated_img, timer, count, cap.count, macht_df, text_scale
            )

            # concatenated_img = cv2.resize(concatenated_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            if args.show:
                # cv2.imshow('field', img_copy)
                # cv2.imshow('layout', img_layout_copy)
                # cv2.imshow('Video', concatenated_img)

                # Добавляем задержку для показа видео в реальном времени
                # if cv2.waitKey(25) & 0xFF == ord("q"):
                #    break
                stframe.image(concatenated_img, channels="BGR")

            if save_output:
                vid_writer.write(concatenated_img)
            df_field.write(
                macht_df.loc[
                    :, ["frame", "x_anchor", "y_anchor", "team", "id", "cls", "conf"]
                ]
            )
            print(
                macht_df.loc[
                    :, ["frame", "x_anchor", "y_anchor", "team", "id", "cls", "conf"]
                ]
            )
            prev_dets = deepcopy(macht_df)
            bd.update_db(macht_df)

        st_prog_bar.progress(
            percent_complete, text=f"Detection in progress ({percent_complete}%)"
        )
        count += 1
    cap.release()
    vid_writer.release()
    bd.close_db()
    df_field.subheader("Данные детекции и трекинга:")
    st_prog_bar.empty()
