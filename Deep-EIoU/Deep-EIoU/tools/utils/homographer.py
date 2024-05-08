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
        layout_img = cv2.imread(self.config['input_layout_image'])
        cap = cv2.VideoCapture(self.config['input_video_path'])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read the video frame.")
            return None, None

        return layout_img, frame

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            concatenated_img, max_width = params
            if x < max_width:  # Clicked on the layout image
                self.points_layout.append((x, y))
            else:  # Clicked on the video frame
                self.points_frame.append((x - max_width, y))
            self.update_display(concatenated_img, max_width)

    def update_display(self, concatenated_img, max_width):
        concatenated_img[:, :] = np.concatenate((self.padded_layout_img, self.padded_first_frame), axis=1)
        for pt_layout in self.points_layout:
            cv2.circle(concatenated_img, pt_layout, 5, (255, 0, 0), -1)
        for pt_frame in self.points_frame:
            cv2.circle(concatenated_img, (pt_frame[0] + max_width, pt_frame[1]), 5, (0, 255, 0), -1)
        for pt_layout, pt_frame in zip(self.points_layout, self.points_frame):
            cv2.line(concatenated_img, pt_layout, (pt_frame[0] + max_width, pt_frame[1]), (0, 0, 255), 2)
        cv2.imshow("Homography Points Selection", concatenated_img)

    def compute_homography_matrix(self):
        if self.config['h_matrix_path'] and os.path.exists(self.config['h_matrix_path']):
            return np.load(self.config['h_matrix_path'])

        self.padded_layout_img, self.padded_first_frame, concatenated_img = self.prepare_images_for_display()
        max_width = max(self.layout_img.shape[1], self.first_frame.shape[1])

        cv2.namedWindow("Homography Points Selection", cv2.WINDOW_NORMAL)
        cv2.imshow("Homography Points Selection", concatenated_img)
        cv2.setMouseCallback("Homography Points Selection", self.click_event, (concatenated_img, max_width))

        print("Instructions:")
        print("- Click corresponding points on the layout image and the video frame.")
        print("- Press 'y' to confirm and calculate homography.")
        print("- Press 'Esc' to quit.")
        print("- Press 'r' to remove the last point match.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key
                cv2.destroyAllWindows()
                return None  # Return early since homography cannot be computed
            elif key == ord('y'):
                if len(self.points_layout) >= 4 and len(self.points_frame) >= 4:
                    H, _ = cv2.findHomography(np.array(self.points_frame), np.array(self.points_layout))
                    if self.config['h_matrix_path']:
                        np.save(self.config['h_matrix_path'], H)
                    cv2.destroyAllWindows()
                    return H
                else:
                    print("Not enough points to compute homography.")
                    cv2.destroyAllWindows()
                    return None  # Return early since homography cannot be computed
            elif key == ord('r') and self.points_layout and self.points_frame:  # Remove the last point match
                self.points_layout.pop()
                self.points_frame.pop()
                self.update_display(concatenated_img, max_width)

    def prepare_images_for_display(self):
        max_height = max(self.layout_img.shape[0], self.first_frame.shape[0])
        max_width = max(self.layout_img.shape[1], self.first_frame.shape[1])

        padded_layout_img = cv2.copyMakeBorder(self.layout_img, 0, max_height - self.layout_img.shape[0], 0, max_width - self.layout_img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_first_frame = cv2.copyMakeBorder(self.first_frame, 0, max_height - self.first_frame.shape[0], 0, max_width - self.first_frame.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

        concatenated_img = np.concatenate((padded_layout_img, padded_first_frame), axis=1)
        return padded_layout_img, padded_first_frame, concatenated_img