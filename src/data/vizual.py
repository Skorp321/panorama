import os
import cv2
import json
import pandas as pd
from tqdm.auto import tqdm

count = 1
imgs_path = "data/interim/keypoints/images/train"

files = os.listdir(imgs_path)


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


for i, file in tqdm(enumerate(files), total=len(files)):
    path_to_image = os.path.join(imgs_path, file)
    anno_path = path_to_image.replace("images", "labels").replace("jpg", "txt")
    img = cv2.imread(path_to_image)
    width = img.shape[1]
    height = img.shape[0]

    anno = pd.read_table(anno_path, header=None, delimiter=' ')
    anno = anno.to_numpy()

    for j in range(0, len(anno[0][5:]), 3):
        x = int(anno[0][j] * width)
        y = int(anno[0][j+1] + height)

        print(x, y)

        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

    os.makedirs('data/test', exist_ok=True)
    cv2.imwrite(f'data/test/file{count}.jpg', img)
    count += 1
