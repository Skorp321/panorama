import os
import cv2
import numpy as np

class HomographySetup:
    def __init__(self, config):
        self.config = config
        self.layout_img, self.first_frame = self.load_and_prepare_images()
        self.points_layout = []
        self.points_frame = []

    def load_and_prepare_images(self):
        layout_img = cv2.imread(self.config.path_to_field)
        cap = cv2.VideoCapture(self.config.path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read the video frame.")
            return None, None

        return layout_img, frame


    def compute_homography_matrix(self):
        if self.config.h_matrix_path and os.path.exists(self.config.h_matrix_path):
            return np.load(self.config.h_matrix_path)

        H, _ = cv2.findHomography(np.array(self.points_frame), np.array(self.points_layout))
        
        if self.config.h_matrix_path:
            np.save(self.config.h_matrix_path, H)
            return H


    def prepare_images_for_display(self, frame, layout):#, layout):
        max_height = max(layout.shape[0], frame.shape[0])
        max_width = max(layout.shape[1], frame.shape[1])
        
        horizontal_margin = int((max_width - layout.shape[1]) / 2.)
        
        padded_layout_img = cv2.copyMakeBorder(layout, 0, 0, horizontal_margin+1, horizontal_margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_first_frame = cv2.copyMakeBorder(frame, 0, max_height - frame.shape[0], 0, max_width - frame.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

        concatenated_img = np.concatenate((padded_layout_img, padded_first_frame), axis=0)
        return padded_layout_img, padded_first_frame, concatenated_img