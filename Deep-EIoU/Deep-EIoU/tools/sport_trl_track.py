import argparse
import os
import sys

import torch
from .utils.stimer import Timer
from .utils.team_classifire import TeamClassifier

import cv2
from tqdm.auto import tqdm
sys.path.append('.')
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
        "--path", default="../../data/Swiss_vs_Slovakia-panoramic_video.mp4", 
        help="path to images or video"
    )
    parser.add_argument(
        "--path_to_det", default="../../models/yolov8m_goalkeeper_1280.pt", 
        help="path to detector model"
    )
    parser.add_argument(
        "--path_to_reid", default="../../models/osnet_ain_x1_0_triplet_custom.pt", 
        help="path to reid model"
    )
    parser.add_argument(
        "--save_path", default="/home/skorp321/Projects/panorama2/data/output", type=str
        )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--benchmark", dest="benchmark", type=str, default='MOT17', 
        help="benchmark to evaluate: MOT17 | MOT20"
        )
    parser.add_argument(
        "--eval", dest="split_to_eval", type=str, default='test', 
        help="split to evaluate: train | val | test"
        )
    parser.add_argument(
        "-f", "--exp_file", default=None, type=str, 
        help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, 
                        help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, 
                        default=None)
    parser.add_argument(
        "--default-parameters", dest="default_parameters", default=False, 
        action="store_true", 
        help="use the default parameters as in the paper")
    parser.add_argument(
        "--save-frames", dest="save_frames", default=False, action="store_true", 
        help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="0", type=str, 
                        help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=1280, type=int, help="test img size")
    parser.add_argument(
        "--fp16", dest="fp16", default=False, action="store_true", 
        help="Adopting mix precision evaluating.")
    parser.add_argument(
        "--fuse", dest="fuse", default=False, action="store_true", 
        help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, 
                        help="tracking confidence threshold")
    parser.add_argument(
        "--track_low_thresh", default=0.1, type=float, 
        help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, 
                        help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, 
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, 
                        help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6, 
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')

    # CMC
    parser.add_argument(
        "--cmc-method", default="file", type=str, 
        help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", 
                        help="use Re-ID flag.")
    parser.add_argument(
        "--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", 
        type=str, help="reid config file path")
    parser.add_argument(
        "--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", 
        type=str, help="reid config file path")
    parser.add_argument(
        '--proximity_thresh', type=float, default=0.5, 
        help='threshold for rejecting low overlap reid matches')
    parser.add_argument(
        '--appearance_thresh', type=float, default=0.25, 
        help='threshold for rejecting low appearance similarity reid matches')

    return parser


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
    
def get_crops(img, crop_width=1280, offset=0):
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
    
def main():
    
    timer = Timer()
    collors = [(0,0,255), (0,255,0), (255,0,0), (125,125,0)]
    args = make_parser().parse_args()
    
    '''if args.trt_file is not None:
        from torch2trt import TRTModule

        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_file))

        x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
        model(x)
        model = model_trt
    else:'''
    model = YOLO(args.path_to_det)
    print(model.names)
    
    model_reid = TeamClassifier(weights_path=args.path_to_reid, model_name='osnet_x1_0')

        # Tracker
    tracker = Deep_EIoU(args, frame_rate=30)
    
    cap = ffmpegcv.VideoCaptureNV(args.path)
    fps = cap.fps
    
    save_path = os.path.join(args.save_path, args.path.split('/')[-1])

    vid_writer = ffmpegcv.VideoWriterNV(save_path, 'h264', fps)
    logger.info(f"save path to video: {save_path}")
    
    text_scale = 2
    size = cap.size
    count = 1
    for frame in tqdm(cap):
        timer.tic()
        img_copy = frame.copy()
        if args.fp16:
            img_copy = img_copy.half()  # to FP16
            
        if count % 1 == 0:
            dets = []
            imgs_list = []
            embed = {'cls':[], 'embs':[]}
            cls_list = []
            
            logger.info(f'Processing seq {count} in {size}')

            imgs, offsets = get_crops(img_copy)
            outputs = model(imgs, imgsz=1280, show_conf=False, show_boxes=False, device=0, stream=False)
            # Process results list
            for offset, result in zip(offsets, outputs):
                boxes = result.boxes  # Boxes object for bounding box outputs
                
                for box in boxes.data:
                    x1, y1 = int(offset + box[0]), int(box[1])
                    x2, y2 = int(offset + box[2]), int(box[3])
                    w = x2 - x1
                    h = y2 - y1
                    cls = int(box[5])
                    collor = collors[cls]
                    cls_list.append(cls)
                    imgs_list.append(img_copy[y1:y2, x1:x2])
                    
                    #embeding = embeding.to('cpu').squeeze()
                    dets.append([cls, x1, y1, x2, y2, 1])
                    
                    #embs.append(embeding)
            embed['cls'] = cls_list      
            embeding = model_reid.extract_features(imgs_list)
            embed['embs'] = embeding
            
            embeddings = pd.DataFrame(embed)
            
            embeddings['team'] = model_reid.classify(embeddings['embs'].to_list(), count)
            arr = np.array(dets, dtype=np.float64)
            team1, team2 = 0, 0
            for i, (cls, emb, team) in embeddings.iterrows():                
                box = arr[i, 1:-1]
                x1,y1,x2,y2 = box
                #print(f'Cls: {cls}')
                if cls != 2:
                    if team == 0 :
                        collor = collors[0]
                        team1 += 1
                    if team == 1:
                        collor = collors[1]                  
                        team2 += 1
                    cv2.rectangle(img_copy, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                collor, 
                                1)
                else:
                    cv2.rectangle(img_copy, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                collors[2], 
                                1)
                    print("Draw ref")
            print(f'Team 1: {team1}, team 2: {team2}')
            detections = arr[:, 1:]
            results = image_track(tracker, detections, embeddings['embs'], args.save_path, args, count)
            timer.toc()
            
            for num_res in results:
                res = num_res.split(',')
                cv2.putText(img_copy, res[1], (int(float(res[2])), int(float(res[3]))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                
            cv2.putText(img_copy, 'frame: %d fps: %.2f num: %d' % (count, 1. / timer.average_time, detections.shape[0]),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            cv2.imshow('Video', img_copy)

            # Добавляем задержку для показа видео в реальном времени
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            count += 1
            vid_writer.write(img_copy)
        else:
            timer.toc()
            cv2.putText(img_copy, 'frame: %d fps: %.2f num: %d' % (count, 1. / timer.average_time, detections.shape[0]),
                        (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            cv2.imshow('Video', img_copy)
            # Добавляем задержку для показа видео в реальном времени
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            count += 1
            vid_writer.write(img_copy)
            
    cap.release()        
    vid_writer.release()
    
if __name__ == "__main__":
    main()