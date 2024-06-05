import json
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
        layout_img = cv2.imread(self.config.input_layout_image)
        cap = cv2.VideoCapture(self.config.path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read the video frame.")
            return None, None

        return layout_img, frame
    
    def get_points_from_layout(self, soccer_field_path_anno: str) -> list:
        sorted_dict = self._extracted_from_compute_homography_matrix_3(
            '/home/skorp321/Projects/panorama/data/soccer_field_anno/annotations/person_keypoints_default.json'
        )
        print(sorted_dict)
        return list(sorted_dict.values())


    def compute_homography_matrix(self):
        if self.config.h_matrix_path and os.path.exists(self.config.h_matrix_path):
            print('Try to finde homography matrixe')
            print(np.load(self.config.h_matrix_path))
        else:
            '''model_keypoint = YOLO(self.config.path_to_keypoints_det)
            outputs = model_keypoint(self.first_frame, half=True, device=0, imgsz=1280)[0]
            field_points = outputs.keypoints.xy.detach().cpu().tolist()
            print(outputs.keypoints)'''
            sorted_dict = self._extracted_from_compute_homography_matrix_3(
                '/home/skorp321/Projects/panorama/data/Swiss_vs_Slovakia-panoramic_video_anno/annotations/person_keypoints_default.json'
            )
            field_points = list(sorted_dict.values())
            layout_points = self.get_points_from_layout(self.config.path_to_field_points)

            H, _ = cv2.findHomography(np.array(field_points), np.array(layout_points))
            print(H)
            self.config.h_matrix_path = '/home/skorp321/Projects/panorama/data/h_matrix_path.npy'
            np.save(self.config.h_matrix_path, H)

        return np.load(self.config.h_matrix_path)

    # TODO Rename this here and in `get_points_from_layout` and `compute_homography_matrix`
    def _extracted_from_compute_homography_matrix_3(self, arg0):
        with open(arg0, 'r') as file:
            data = file.read()
        data_dict = json.loads(data)
        categoris = data_dict['categories'][0]['keypoints']
        df = data_dict['annotations'][0]['keypoints']
        res = {data: df[i * 3:(i + 1) * 3 - 1] for i, data in enumerate(categoris)}
        sorted_list = sorted(res.items(), key=lambda x: x[0])
        return dict(sorted_list)


    def prepare_images_for_display(self, frame, layout):#, layout):
        max_height = max(layout.shape[0], frame.shape[0])
        max_width = max(layout.shape[1], frame.shape[1])
        
        horizontal_margin = int((max_width - layout.shape[1]) / 2.)
        
        padded_layout_img = cv2.copyMakeBorder(layout, 0, 0, horizontal_margin+1, horizontal_margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_first_frame = cv2.copyMakeBorder(frame, 0, max_height - frame.shape[0], 0, max_width - frame.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

        concatenated_img = np.concatenate((padded_layout_img, padded_first_frame), axis=0)
        return padded_layout_img, padded_first_frame, concatenated_img