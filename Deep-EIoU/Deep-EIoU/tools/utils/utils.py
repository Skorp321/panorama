import os

import cv2
from loguru import logger
from stimer import Timer

trackerTimer = Timer()
timer = Timer()

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def get_crops(img, crop_width=1280, offset=30):
    """
    Generate crops from an image.
    Args:
        img (numpy.ndarray): The input image.
        crop_width (int, optional): The width of each crop. Defaults to 1280.
        offset (int, optional): The offset for each crop. Defaults to 0.
    Returns:
        Tuple[List[numpy.ndarray], List[int]]: A tuple containing a list of 
        cropped images and a list of corresponding coordinate offsets.
    """
    orig_width = img.shape[1]
    crops = []
    coord_offsets = []
    for c_idx in range(0, orig_width, crop_width):
        w_start = max(0, c_idx - offset)
        w_end = min(c_idx + crop_width, orig_width)
        if w_end <= w_start:
            continue
        crop = img[:, w_start:w_end]
        coord_offsets.append(w_start)
        crops.append(crop)

    return crops, coord_offsets


def image_track(tracker, detections, embeddings, sct_output_path, args, frame_id):
    
    results = []
    
    num_frames = len(detections)

    scale = min(1440/1280, 800/720)

    det = detections
    embs = embeddings
    
    if det is not None:

        '''embs = [e[0] for e in embs]
        embs = np.array(embs)'''
        
        trackerTimer.tic()
        online_targets = tracker.update(det, embs)
        trackerTimer.toc()

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.last_tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            vertical = False
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

                # save results
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
        timer.toc()

    else:
        timer.toc()
    if frame_id % 100 == 0:
        logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    save_path = os.path.join(sct_output_path, 'result.txt')
    with open(save_path, 'w') as f:
        f.writelines(results)
    logger.info(f"save SCT results to {sct_output_path}")
    return results


def apply_homography_to_point(point, H):
    if point is None:
        return point
    point = np.array(point)

    # Convert point to homogeneous coordinates
    point_homogeneous = np.append(point, 1)

    # Apply the homography matrix
    #point_transformed_homogeneous = np.dot(H, point_homogeneous)
    point_transformed_homogeneous = np.matmul(H, np.transpose(point_homogeneous))
    # Convert back to Cartesian coordinates
    point_transformed = point_transformed_homogeneous[:2] / point_transformed_homogeneous[2]
    return [int(np.abs(value)) for value in point_transformed]
    

def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU For Evaluation!")

    parser.add_argument(
        "--path",
        default="/home/skorp321/Projects/panorama/data/Swiss_vs_Slovakia-panoramic_video.mp4",
        help="path to images or video",
    )
    parser.add_argument(
        "--show",
        default=True,
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
    parser.add_argument(
        "--init_tresh",
        default=3,
        help="frames to initialize the algorithms.",
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
        "--track_buffer", type=int, default=160, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.7,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--min_box_area", type=float, default=4, help="filter out tiny boxes"
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

def iou(box1, box2):
    """
    Compute the Intersection Over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : ndarray
        (x1, y1, x2, y2) of the first box.
    box2 : ndarray
        (x1, y1, x2, y2) of the second box.

    Returns
    -------
    float
        in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def non_max_suppression(boxes, iou_threshold):
    """
    Perform non-maximum suppression, given a list of boxes with their
    corresponding scores and class labels.

    Parameters
    ----------
    boxes : list of lists or 2D ndarray
        List with elements [x1, y1, x2, y2, score, class_label].
    iou_threshold : float
        Threshold for IoU to determine whether boxes overlap too much.

    Returns
    -------
    list of lists
        Filtered boxes after NMS.
    """
    boxes = boxes.to_numpy()
    new_boxes = boxes.copy()
    
    new_boxes[:,:-1] = boxes[:,1:]
    new_boxes[:,-1] = boxes[:,0]
    boxes = new_boxes

    if len(boxes) == 0:
        return []

    # Sort the boxes by score in descending order
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    picked_boxes = []

    while boxes:
        # Pick the box with the highest score
        current_box = boxes.pop(0)
        picked_boxes.append(current_box)

        # Compare the picked box with the rest of the boxes
        boxes = [box for box in boxes if box[5] != current_box[5] or iou(current_box, box) <= iou_threshold]

    return np.array(picked_boxes)


def draw_annos(collor, img_copy, img_layout_copy, x1, y1, x2, y2, row, h_point):
    cv2.circle(img_layout_copy, h_point, 6, collor, -1)
    cv2.putText(img_layout_copy, 
                                str(int(row['id'])), 
                                h_point, 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                collor, 
                                1, 
                                cv2.LINE_AA)
                    
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), collor, 1)
    cv2.putText(
                            img_copy,
                            str(int(row['id'])),
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            collor,
                            1,
                            cv2.LINE_AA,
                        )