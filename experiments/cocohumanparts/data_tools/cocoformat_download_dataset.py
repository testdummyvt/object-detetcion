import os
import argparse
import yaml
from pathlib import Path
import shutil
from ultralytics.utils.downloads import download


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

    # URLs for the training and validation images
    images_train_url = remote_dataset_url + "/images/train.zip"
    images_val_url = remote_dataset_url + "/images/val.zip"

    # Download images in parallel using 2 threads
    download([images_train_url, images_val_url], local_dataset_dir, threads=2)

    # URLs for the training and validation labels (COCO format)
    labels_train_url = (
        remote_dataset_url + "/person_humanparts_train2017_coco_format.json"
    )
    labels_val_url = remote_dataset_url + "/person_humanparts_val2017_coco_format.json"

    # Download labels in parallel using 2 threads
    download([labels_train_url, labels_val_url], local_dataset_dir, threads=2)

    download([labels_train_url], local_dataset_dir, threads=1)

    # Rename val folder to valid
    os.rename(
        os.path.join(local_dataset_dir, "val"), os.path.join(local_dataset_dir, "valid")
    )

    # Copy label files to train and valid folders
    current_train_json_file = (
        local_dataset_dir + "/" + "person_humanparts_train2017_coco_format.json"
    )
    current_valid_json_file = (
        local_dataset_dir + "/" + "person_humanparts_val2017_coco_format.json"
    )

    new_train_json_file = (
        local_dataset_dir + "/" + "train" + "/" + "_annotations.coco.json"
    )
    new_valid_json_file = (
        local_dataset_dir + "/" + "valid" + "/" + "_annotations.coco.json"
    )

    shutil.copyfile(current_train_json_file, new_train_json_file)
    shutil.copyfile(current_valid_json_file, new_valid_json_file)


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
