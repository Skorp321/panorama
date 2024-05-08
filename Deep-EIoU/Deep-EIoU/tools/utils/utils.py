import os

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