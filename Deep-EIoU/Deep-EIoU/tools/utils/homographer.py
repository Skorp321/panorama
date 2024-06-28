import json
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import ffmpegcv
from loguru import logger


class HomographySetup:
    def __init__(self, config):
        self.config = config
        self.layout_img, self.first_frame, self.points_names = self.load_and_prepare_images()
        self.points_layout = []
        self.points_frame = []
        self.data = pd.DataFrame()

    def load_and_prepare_images(self):
        layout_img = cv2.imread(self.config.path_to_field)
        cap = ffmpegcv.VideoCapture(self.config.path_to_field)
        with open(self.config.path_to_field_points) as file:
            keypoints_field = json.load(file)

        names = keypoints_field["categories"][0]["keypoints"]
        
        if not cap.isOpened():
            print("Error: Could not open file.")
            return None, None

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the video frame.")
            return None, None

        cap.release()

        return layout_img, frame, names

    def compute_homography_matrix(self):
        model_keypoint = YOLO(self.config.path_to_keypoints_det)
        outputs = model_keypoint(self.first_frame, half=True, device=0, imgsz=1280)[0]
        keypoints = outputs.keypoints.data[0].detach().cpu().numpy()
        
        layout_points = self._extracted_from_compute_homography_matrix_3(
            self.config.path_to_field_points
        )
        
        df = pd.DataFrame(keypoints, index=self.names, columns=["x", "y", "conf"])
        df_n = df[df["conf"] >= 0.75]
        field_points = layout_points[layout_points.index.isin(df_n.index)]

        H, _ = cv2.findHomography(df_n.values, field_points.values)

        '''self.config.h_matrix_path = "/container_dir/panorama/data/h_matrix_path.npy"
        np.save(self.config.h_matrix_path, H)'''

        return H

    # TODO Rename this here and in `get_points_from_layout` and `compute_homography_matrix`
    def _extracted_from_compute_homography_matrix_3(self, path_to_field):
        with open(path_to_field) as file:
            keypoints_map_pos = json.load(file)
        field_points = np.array(keypoints_map_pos["annotations"][0]["keypoints"]).reshape(-1, 3)

        field_df = pd.DataFrame(
            field_points,
            index=self.names,
            columns=["x", "y", "viz"])
        return field_df

    def prepare_images_for_display(self, frame, layout):
        logger.info(f"layout:{layout.shape}, frame: {frame.shape}")
        max_height = max(layout.shape[0], frame.shape[0])
        max_width = max(layout.shape[1], frame.shape[1])

        horizontal_margin = int((max_width - layout.shape[1]) / 2.0)

        padded_layout_img = cv2.copyMakeBorder(
            layout,
            0,
            0,
            horizontal_margin + 1,
            horizontal_margin,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        padded_first_frame = cv2.copyMakeBorder(
            frame,
            0,
            max_height - frame.shape[0],
            0,
            max_width - frame.shape[1],
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        logger.info(
            f"layout:{padded_layout_img.shape}, frame: {padded_first_frame.shape}"
        )
        concatenated_img = np.concatenate(
            (padded_layout_img, padded_first_frame), axis=0
        )
        logger.info(f"concatenated_img:{concatenated_img.shape}")
        return padded_layout_img, padded_first_frame, concatenated_img
