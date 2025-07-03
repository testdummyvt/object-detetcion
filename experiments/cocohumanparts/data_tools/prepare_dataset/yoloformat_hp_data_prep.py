import os
import json
import numpy as np
from tqdm import tqdm
import shutil
from typing import Dict, List, Tuple


def sort_labels_by_image_id(labels_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    This function sorts the labels by image_id.

    Args:
        labels_list (List[Dict]): A list of dictionaries containing annotations.

    Returns:
        Dict[str, List[Dict]]: A dictionary with image_id as keys and corresponding labels as values.

    """
    images_labels_dict: Dict[str, List[Dict]] = {}
    for i, labels_dict in tqdm(
        enumerate(labels_list), desc="Sorting labels by image_id"
    ):
        image_id = str(labels_dict["image_id"])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


def normalize_bbox(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    This function normalizes the bounding box coordinates to be in [0, 1] range.

    Args:
        bbox (Tuple[float, float, float, float]): A tuple of (x, y, width, height).
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        Tuple[float, float, float, float]: A tuple of normalized (x_center, y_center, width, height).

    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def convert_pbbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    This function corrects and converts part bounding box to normalized YOLO format.

    Args:
        bbox (List[int]): A list of [x, y, w, h].
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        Tuple[float, float, float, float]: A tuple of normalized (x_center, y_center, width, height).

    """
    if bbox[0] + bbox[2] > img_w:
        bbox[2] = img_w - 1 - bbox[0]
    if bbox[1] + bbox[3] > img_h:
        bbox[3] = img_h - 1 - bbox[1]
    if bbox[0] < 0:
        bbox[2] += bbox[0]
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[3] += bbox[1]
        bbox[1] = 0

    return normalize_bbox(bbox, img_w, img_h)


def convert_hpbbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    This function corrects and converts human part bounding box to normalized YOLO format.

    Args:
        bbox (List[int]): A list of [x_min, y_min, x_max, y_max].
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        Tuple[float, float, float, float]: A tuple of normalized (x_center, y_center, width, height).

    """
    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    if bbox[0] + bbox[2] > img_w:
        bbox[2] = img_w - 1 - bbox[0]
    if bbox[1] + bbox[3] > img_h:
        bbox[3] = img_h - 1 - bbox[1]
    if bbox[0] < 0:
        bbox[2] += bbox[0]
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[3] += bbox[1]
        bbox[1] = 0

    return normalize_bbox(bbox, img_w, img_h)


def process_coco_human_parts(
    anno_data_file: str, images_dir: str, dst_images_dir: str, dst_labels_dir: str
) -> None:
    """
    This function processes the COCO human parts annotations to be in YOLO format and saves them.

    Args:
        anno_data_file (str): Path to COCO human parts annotation file.
        images_dir (str): Directory containing the image files.
        dst_images_dir (str): Directory where images will be copied.
        dst_labels_dir (str): Directory where labels will be saved.

    Returns:
        None

    """
    with open(anno_data_file, "r") as f:
        anno_data = json.load(f)
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    images_anno_data = sort_labels_by_image_id(anno_data["annotations"])
    images_info = anno_data["images"]

    for image_info in tqdm(images_info, desc="Processing images"):
        img_id = image_info["id"]
        img_w, img_h = image_info["width"], image_info["height"]

        img_file_name = image_info["file_name"]
        img_ext = img_file_name.split(".")[-1]
        img_file_path = os.path.join(images_dir, img_file_name)
        dst_img_file_path = os.path.join(dst_images_dir, img_file_name)
        dst_label_path = os.path.join(
            dst_labels_dir, img_file_name.replace(img_ext, "txt")
        )

        person_txt_list: List[List[float]] = []
        for person_anno in images_anno_data[str(img_id)]:
            person_bbox = convert_pbbox_to_yolo(person_anno["bbox"], img_w, img_h)
            person_txt_list.append([0] + list(person_bbox))

            for part_label in range(1, 7):
                hier_index = (part_label - 1) * 5
                part_bbox = person_anno["hier"][hier_index : hier_index + 4]
                part_ignore = person_anno["hier"][hier_index + 4]
                if part_ignore != 0:
                    yolo_part_bbox = convert_hpbbox_to_yolo(part_bbox, img_w, img_h)
                    person_txt_list.append([part_label] + list(yolo_part_bbox))
            text_string = ""
            for row in person_txt_list:
                text_string += " ".join(map(str, row)) + "\n"

        shutil.copyfile(img_file_path, dst_img_file_path)
        with open(dst_label_path, "w") as f:
            f.write(text_string)


if __name__ == "__main__":
    val_image_dir = "C:\\local\\datasets\\mscoco\\coco\\images\\val2017"
    train_image_dir = "C:\\local\\datasets\\mscoco\\coco\\images\\train2017"

    val_anno_file = "C:\\local\\datasets\\mscoco\\person_humanparts_val2017.json"
    train_anno_file = "C:\\local\\datasets\\mscoco\\person_humanparts_train2017.json"

    dst_dir = "C:\\local\\datasets\\cocohumanparts"

    dst_val_image_dir = os.path.join(dst_dir, "images", "val")
    dst_train_image_dir = os.path.join(dst_dir, "images", "train")
    dst_val_label_dir = os.path.join(dst_dir, "labels", "val")
    dst_train_label_dir = os.path.join(dst_dir, "labels", "train")
    # process train data
    process_coco_human_parts(
        train_anno_file, train_image_dir, dst_train_image_dir, dst_train_label_dir
    )

    # process val data
    process_coco_human_parts(
        val_anno_file, val_image_dir, dst_val_image_dir, dst_val_label_dir
    )
