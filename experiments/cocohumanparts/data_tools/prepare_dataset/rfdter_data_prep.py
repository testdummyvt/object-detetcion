import os
import json
import yaml
from PIL import Image
import shutil
from glob import glob


# Function to convert YOLO bbox to COCO bbox
def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = map(float, yolo_bbox)
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    return [x_min, y_min, width, height]


if __name__ == "__main__":
    dataset_name = "cocohumanparts"
    # Paths to your Ultralytics dataset
    ultralytics_root = (
        "/home/ubuntu/data/cocohumanparts"  # Replace with your ultralytics folder
    )
    # Target dataset paths
    target_root = "/home/ubuntu/data/cocohumanparts_rf"  # Replace with where you want the new dataset

    splits = ["train", "val"]  # "test"
    target_splits = ["train", "valid"]  # "test"

    data_yaml_path = os.path.join(ultralytics_root, "data.yaml")

    # Load the data.yaml file to get class names
    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    class_names = list(data_yaml["names"].values())
    num_classes = len(class_names)

    # Create the target directory structure
    for split in target_splits:
        os.makedirs(os.path.join(target_root, split), exist_ok=True)

    # Process each split (train, valid, test)
    for split, target_split in zip(splits, target_splits):
        print(f"Processing {split} split...")

        # Initialize COCO JSON structure
        coco_data = {
            "info": {"description": f"{dataset_name} {target_split} dataset"},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Add categories
        for class_id, class_name in enumerate(class_names):
            coco_data["categories"].append(
                {
                    "id": class_id,  # + 1,  # COCO IDs start at 1
                    "name": class_name,
                    "supercategory": "none",
                }
            )

        # Paths for images and labels
        image_dir = os.path.join(ultralytics_root, "images", split)
        label_dir = os.path.join(ultralytics_root, "labels", split)
        target_image_dir = os.path.join(target_root, target_split)

        image_id = 0
        annotation_id = 0

        # Process each image
        for src_image_path in glob(image_dir + os.sep + "*.*"):
            if not src_image_path.endswith((".jpg", ".jpeg", ".png")):
                continue

            dst_image_path = (
                src_image_path.replace(os.sep + "images", "")
                .replace(ultralytics_root, target_root)
                .replace(split, target_split)
            )

            print(src_image_path, dst_image_path)
            shutil.copy(src_image_path, dst_image_path)

            # Get image dimensions
            with Image.open(src_image_path) as img:
                img_width, img_height = img.size

            # Add image to COCO JSON
            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": os.path.basename(src_image_path),
                    "width": img_width,
                    "height": img_height,
                }
            )

            # Load corresponding label file
            label_path = src_image_path.replace("images", "labels")
            label_path = os.path.splitext(label_path)[0] + ".txt"

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue  # Skip invalid lines
                        class_id, x_center, y_center, width, height = parts[:5]
                        class_id = int(class_id)  # + 1  # COCO class IDs start at 1

                        # Convert YOLO bbox to COCO bbox
                        bbox = yolo_to_coco_bbox(
                            [x_center, y_center, width, height], img_width, img_height
                        )
                        area = bbox[2] * bbox[3]  # width * height

                        # Add annotation to COCO JSON
                        coco_data["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "bbox": bbox,
                                "area": area,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1

            image_id += 1

        # Save COCO JSON
        coco_json_path = os.path.join(target_image_dir, "_annotations.coco.json")
        with open(coco_json_path, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"Finished processing {split} split. Saved to {coco_json_path}")

    print("Conversion complete!")
