import argparse
import os
from random import shuffle
import shutil
import cv2
import json
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy
import ffmpegcv
import yaml


def make_parser():
    parser = argparse.ArgumentParser(description="Create keypoint dataset")
    parser.add_argument(
        "--input_video_folder",
        default="/container_dir/data/raw/video",
        type=str,
        help="Path to input video",
    )
    parser.add_argument(
        "--output_dir",
        default="/container_dir/data/interim",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument("--fps", type=int, default=25, help="FPS of output video")
    # parser.add_argument('--json_path', type=str, default='data/Swiss_vs_Slovakia-panoramic_video_anno/annotations/person_keypoints_default.json', help='Path to anno json file')

    return parser


def json_to_pandas(json_path: str) -> pd.DataFrame:
    """Convert json file to pandas dataframe"""

    if not os.path.exists(json_path):
        raise ValueError("File does not exist")

    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = pd.DataFrame(data["annotations"])
    categories = pd.DataFrame(data["categories"])
    images = pd.DataFrame(data["images"])

    return annotations, categories, images


def create_keypoint_detaset(json_path: str) -> pd.DataFrame:
    """
    This function aims to create a DataFrame from a JSON file containing annotations, categories, and
    images data, with specific columns for image ID, keypoints, bounding box, and annotation keypoints
    names.

    :param json_path: The `json_path` parameter in the `create_keypoint_dataset` function is a string
    that represents the file path to a JSON file containing annotations, categories, and images data.
    This function is designed to read the JSON file, extract relevant information such as keypoints,
    image IDs, and bounding boxes,
    :type json_path: str
    :return: A DataFrame containing image IDs, keypoints, bounding boxes, and annotation keypoints is
    being returned.
    """

    dataframe = pd.DataFrame()
    dataframe_id = []
    dataframe_keypoints = []
    dataframe_bbox = []

    annotations, categories, images = json_to_pandas(json_path)
    keypoints_names = categories["keypoints"]

    for _, annotation in annotations.iterrows():
        dataframe_id.append(annotation["id"])
        dataframe_keypoints.append(annotation["keypoints"])
        dataframe_bbox.append(annotation["bbox"])

    dataframe["id"] = dataframe_id
    dataframe["keypoints"] = dataframe_keypoints
    dataframe["bbox"] = dataframe_bbox

    return dataframe, keypoints_names


def process_video(args, jj, full_path) -> None:

    i = 1
    print(f"Process {jj} video!")
    os.makedirs(os.path.join(args.output_dir, f"folder{jj}", "images"), exist_ok=True)
    os.makedirs(
        os.path.join(args.output_dir, f"folder{jj}", "annotations"), exist_ok=True
    )
    cap = ffmpegcv.VideoCapture(full_path)
    width = cap.width
    height = cap.height

    anno_start = os.path.split(full_path)[0].replace("video", "cvat")
    anno_folder = os.path.join(anno_start, os.path.split(full_path)[-1])
    video_name = anno_folder.split(".")[0]
    anno_path = os.path.join(video_name, "annotations", "person_keypoints_default.json")
    anno_data, keypoints_names = create_keypoint_detaset(anno_path)

    with tqdm(total=cap.count) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(
                os.path.join(args.output_dir, f"folder{jj}", "images", f"{i:04d}.jpg"),
                frame,
            )

            res_anno = [0]
            bbox = []
            keypoints = [[]]
            keypoints = anno_data[anno_data["id"] == i]["keypoints"].tolist()
            bbox = anno_data[anno_data["id"] == i]["bbox"].tolist()
            bbox_xcyc = deepcopy(bbox)

            bbox_xcyc[0][0] = bbox[0][0] + ((bbox[0][2] - bbox[0][0]) / 2.0)
            bbox_xcyc[0][1] = bbox[0][1] + ((bbox[0][3] - bbox[0][1]) / 2.0)
            bbox_xcyc[0][2] = bbox[0][2] - bbox[0][0]
            bbox_xcyc[0][3] = bbox[0][3] - bbox[0][1]

            for j in range(0, len(bbox_xcyc[0]), 2):
                bbox_xcyc[0][j] = bbox_xcyc[0][j] / width
                bbox_xcyc[0][j + 1] = bbox_xcyc[0][j + 1] / height
            bbox_norm = bbox_xcyc[0].copy()

            for j in range(0, len(keypoints[0]), 3):
                keypoints[0][j] = keypoints[0][j] / width
                if keypoints[0][j] > 1:
                    keypoints[0][j] = 1.0
                elif keypoints[0][j] < 0:
                    keypoints[0][j] = 0.0
                keypoints[0][j + 1] = keypoints[0][j + 1] / height
                if keypoints[0][j + 1] > 1:
                    keypoints[0][j + 1] = 1.0
                elif keypoints[0][j + 1] < 0:
                    keypoints[0][j + 1] = 0.0
            keypoints_norm = keypoints[0].copy()

            [res_anno.append(val) for val in bbox_norm]
            [res_anno.append(keypoint) for keypoint in keypoints_norm]
            df = pd.DataFrame(res_anno)
            # Преобразуем DataFrame в JSON строку
            df_str = df.to_csv(index=False, sep=" ", header=False)
            df_str = df_str.replace("\n", " ")
            os.makedirs(
                os.path.join(args.output_dir, f"folder{jj}", "labels"), exist_ok=True
            )
            # Записываем JSON строку в файл
            with open(
                os.path.join(args.output_dir, f"folder{jj}", "labels", f"{i:04d}.txt"),
                "w",
            ) as file:
                file.write(df_str)
            # df.to_csv(os.path.join(args.output_dir, 'annotations', f'{i:04d}.txt'), sep=',', line_terminator=' ', index=False)

            i += 1
            pbar.update(1)


def copy_files(files: list, kind: str):
    print(f"Copying {kind} files...")
    count = 1
    for file in tqdm(files):
        # full_img_path = os.path.join(args.output_dir, "images", file)
        full_anno_path = file.replace(".jpg", ".txt").replace("images", "labels")

        dst_img_dir = os.path.join(args.output_dir, "keypoints", "images", kind)
        os.makedirs(dst_img_dir, exist_ok=True)
        dst_anno_dir = dst_img_dir.replace("images", "labels")
        os.makedirs(dst_anno_dir, exist_ok=True)

        dst_img = os.path.join(dst_img_dir, f"file_{count:04d}.jpg")
        dst_anno = dst_img.replace(".jpg", ".txt").replace("images", "labels")

        shutil.copyfile(file, dst_img)
        shutil.copyfile(full_anno_path, dst_anno)
        count += 1


"""def create_yolo_kaypoints_dataset(args, j, full_path):

    process_video(args, j, full_path)

    files = os.listdir(os.path.join(args.output_dir, "images"))
    len_file = len(files)
    files = shuffle(files)
    train_files = files[: int(len_file * 0.8)]
    val_files = files[int(len_file * 0.8) :]

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    shutil.rmtree(os.path.join(args.output_dir, "images"))
    shutil.rmtree(os.path.join(args.output_dir, "annotations"))"""


def generate_yaml_file(args):
    print("Generating YAML file...")
    yaml_file = os.path.join(args.output_dir, "keypoints", "data.yaml")

    data = {}
    data["path"] = args.output_dir + "/keypoints"
    data["train"] = "images/train"
    data["val"] = "images/val"
    annotations, categories, _ = json_to_pandas(
        "/container_dir/data/raw/cvat/Swiss_vs_slovakia-Panorama/annotations/person_keypoints_default.json"
    )
    num_keypoints = len(categories["keypoints"][0])
    data["flip_idx"] = list(range(28))
    data["kpt_shape"] = [num_keypoints, 3]
    data["names"] = {0: "field"}
    data["keypoints"] = [
        {"name": str(x), "id": int(y)} for y, x in enumerate(categories["keypoints"][0])
    ]
    # df = pd.DataFrame(data)
    # json_string = df.to_json(orient="records", lines=False)

    with open(yaml_file, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":

    args = make_parser().parse_args()

    videos = os.listdir(args.input_video_folder)

    for j, video in enumerate(videos):
        full_path = os.path.join(args.input_video_folder, video)

        process_video(args, j, full_path)

    folders = os.listdir(os.path.join(args.output_dir))
    all_files = []
    for folder in folders:
        files = os.listdir(os.path.join(args.output_dir, folder, "images"))
        for file in files:
            all_files.append(os.path.join(args.output_dir, folder, "images", file))
    len_file = len(all_files)
    shuffle(all_files)
    train_files = all_files[: int(len_file * 0.8)]
    val_files = all_files[int(len_file * 0.8) :]

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    for folder in folders:
        shutil.rmtree(os.path.join(args.output_dir, folder))

    generate_yaml_file(args)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    

