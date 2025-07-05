import os
from glob import glob
import json
import shutil
import yaml

categories = [
    {"id": 1, "supercategory": "person", "name": "person"},
    {"id": 2, "supercategory": "head", "name": "head"},
    {"id": 3, "supercategory": "face", "name": "face"},
    {"id": 4, "supercategory": "hand", "name": "hand"},
    {"id": 5, "supercategory": "foot", "name": "foot"},
]




def process_coco_human_parts(anno_data_file: str) -> None:
    """
    Read COCO label file and modify class values:
    - Class 5 becomes 4
    - Classes 6 and 7 become 5
    """
    with open(anno_data_file, "r") as f:
        anno_data = json.load(f)

    backup_anno_file = anno_data_file.replace(".json", "_backup.json")
    shutil.copyfile(anno_data_file, backup_anno_file)
    print(f"Backup created at {backup_anno_file}")

    new_anno_data = {
        "categories": categories,
        "annotations": [],
        "images": anno_data["images"],
    }

    new_annos = []
    for anno in anno_data["annotations"]:
        class_id = anno["category_id"]
        if class_id == 5:
            class_id = 4
        elif class_id in [6, 7]:
            class_id = 5
        anno["category_id"] = class_id
        new_annos.append(anno)

    new_anno_data["annotations"] = new_annos

    with open(anno_data_file, "w") as f:
        json.dump(new_anno_data, f)


def yoloformat_process_yolo_labels(input_file):
    """
    Read YOLO label file and modify class values:
    - Class 4 becomes 3
    - Classes 5 and 6 become 4
    """

    with open(input_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if not line.strip():
            continue

        try:
            values = line.strip().split(" ")
            if len(values) != 5:
                print(f"Warning: Invalid format in line: {line.strip()}")
                continue

            class_id = int(values[0])
            # Apply class mapping
            if class_id == 4:
                new_class_id = 3
            elif class_id in [5, 6]:
                new_class_id = 4
            else:
                new_class_id = class_id  # Keep other classes unchanged

            x1, y1, x2, y2 = values[1:]
            new_line = f"{new_class_id} {x1} {y1} {x2} {y2}\n"
            modified_lines.append(new_line)

        except ValueError:
            print(f"Warning: Invalid number format in line: {line.strip()}")
            continue

    with open(input_file, "w") as f:
        f.writelines(modified_lines)


def yoloformat_process_dir(input_dir):
    # Make a backup of the original folder
    backup_dir = input_dir + "_backup"

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(input_dir, backup_dir)
    print(f"Backup created at {backup_dir}")

    files = glob(os.path.join(input_dir, "*.txt"))
    for input_path in files:
        yoloformat_process_yolo_labels(input_path)


def update_data_yaml(local_dataset_dir: str) -> None:
    data_yaml_path = os.path.join(local_dataset_dir, "data.yaml")

    # Load the existing data.yaml file
    with open(data_yaml_path, "r") as data_file:
        data = yaml.safe_load(data_file)

    backup_anno_file = data_yaml_path.replace(".yaml", "_backup.yaml")
    shutil.copyfile(data_yaml_path, backup_anno_file)
    print(f"Backup created at {backup_anno_file}")

    # Update the data.yaml file with the new local dataset directory
    data["names"] = {0: "person", 1: "head", 2: "face", 3: "hand", 4: "foot"}

    # Save the updated data.yaml file
    with open(data_yaml_path, "w") as updated_data_file:
        yaml.dump(data, updated_data_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process data folder based on format type."
    )
    parser.add_argument("--datafolder", type=str, help="Path to the data folder")
    parser.add_argument(
        "--format_type",
        type=str,
        choices=["coco", "yolo"],
        help="Format type: coco or yolo",
    )

    args = parser.parse_args()

    datafolder = args.datafolder
    format_type = args.format_type

    if format_type == "coco":
        files = glob(os.path.join(datafolder, "*", "*.json"))
        for input_path in files:
            process_coco_human_parts(input_path)
    elif format_type == "yolo":
        update_data_yaml(datafolder)
        folders = glob(os.path.join(datafolder, "labels", "*"))
        for f in folders:
            # check if it is a directory
            if os.path.isdir(f):
                yoloformat_process_dir(f)
