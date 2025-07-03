import os
import argparse
import yaml
from pathlib import Path
from od_engine.utils.download import download


def update_data_yaml(local_dataset_dir: str) -> None:
    """
    Update the path in the given data.yaml file to the specified local dataset directory.

    Args:
        local_dataset_dir (str): The local directory path to the dataset.
    """

    data_yaml_path = os.path.join(local_dataset_dir, "data.yaml")

    # Load the existing data.yaml file
    with open(data_yaml_path, "r") as data_file:
        data = yaml.safe_load(data_file)

    # Update the data.yaml file with the new local dataset directory
    data["path"] = os.path.abspath(local_dataset_dir)

    # Save the updated data.yaml file
    with open(data_yaml_path, "w") as updated_data_file:
        yaml.dump(data, updated_data_file)


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

    # Download data.yaml
    data_yaml_url = remote_dataset_url + "/data.yaml"
    download(data_yaml_url, local_dataset_dir)

    # Update data.yaml file with current local dataset directory
    update_data_yaml(local_dataset_dir)

    # URLs for the training and validation labels
    labels_train_url = remote_dataset_url + "/labels/train.zip"
    labels_val_url = remote_dataset_url + "/labels/val.zip"

    # Download labels in parallel using 2 threads
    download(
        [labels_train_url, labels_val_url],
        local_dataset_dir + os.sep + "labels",
        threads=2,
    )

    # URLs for the training and validation images
    images_train_url = remote_dataset_url + "/images/train.zip"
    images_val_url = remote_dataset_url + "/images/val.zip"

    # Download images in parallel using 2 threads
    download(
        [images_train_url, images_val_url],
        local_dataset_dir + os.sep + "images",
        threads=2,
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
