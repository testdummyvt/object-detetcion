import os
import argparse
import yaml
from pathlib import Path
import shutil
from od_engine.utils.download import download


def main(local_dataset_dir):
    """
    This function downloads the COCO Human Parts dataset and its corresponding data.yaml file
    from a remote URL to a local directory.

    Args:
        local_dataset_dir (str): The local directory to store the downloaded dataset. Defaults to the current directory.
    """
    # Remote URL for the dataset and data.yaml file
    remote_dataset_url = (
        "https://huggingface.co/datasets/testdummyvt/cocohumanparts/resolve/main"
    )
    # Create the local dataset directory if it does not exist
    os.makedirs(local_dataset_dir, exist_ok=True)

    # Create local images and annotations directories
    local_images_dir = os.path.join(local_dataset_dir, "images")
    local_annotations_dir = os.path.join(local_dataset_dir, "annotations")
    os.makedirs(local_images_dir, exist_ok=True)
    os.makedirs(local_annotations_dir, exist_ok=True)

    # URLs for the training and validation images
    images_train_url = remote_dataset_url + "/images/train.zip"
    images_val_url = remote_dataset_url + "/images/val.zip"

    # Download images in parallel using 2 threads
    download([images_train_url, images_val_url], local_images_dir, threads=2)

    # URLs for the training and validation labels (COCO format)
    labels_train_url = (
        remote_dataset_url + "/annotations/person_humanparts_train2017_coco_format.json"
    )
    labels_val_url = (
        remote_dataset_url + "/annotations/person_humanparts_val2017_coco_format.json"
    )

    # Download labels in parallel using 2 threads
    download([labels_train_url, labels_val_url], local_annotations_dir, threads=2)

    # Rename the files to train.json and valid.json
    os.rename(
        os.path.join(
            local_annotations_dir, "person_humanparts_train2017_coco_format.json"
        ),
        os.path.join(local_annotations_dir, "instances_train.json"),
    )
    os.rename(
        os.path.join(
            local_annotations_dir, "person_humanparts_val2017_coco_format.json"
        ),
        os.path.join(local_annotations_dir, "instances_val.json"),
    )
    # Copy validation to test
    shutil.copyfile(
        os.path.join(local_annotations_dir, "instances_val.json"),
        os.path.join(local_annotations_dir, "instances_test.json"),
    )


if __name__ == "__main__":
    # Setup argument parser with one optional argument for the local dataset directory
    parser = argparse.ArgumentParser(description="Download COCO Human Parts dataset.")
    parser.add_argument(
        "--local_dataset_dir",
        type=str,
        default=".",
        help="The local directory to store the downloaded dataset.",
    )

    # Parse the argument and get the local dataset directory
    args = parser.parse_args()
    local_dataset_dir = args.local_dataset_dir

    # Call the main function with the specified local dataset directory
    main(local_dataset_dir)
