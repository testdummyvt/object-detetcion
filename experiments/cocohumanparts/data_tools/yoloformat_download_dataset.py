import os
import argparse
import yaml
from pathlib import Path
from torch.hub import download_url_to_file
import zipfile
from concurrent.futures import ThreadPoolExecutor


def download(url, dir=".", unzip=True, delete=True, threads=1):
    """
    Downloads and unzips files from a URL or list of URLs using PyTorch Hub.

    Args:
        url (str or list): The URL or list of URLs to download from.
        dir (str, optional): The directory to save the files in. Defaults to ".".
        unzip (bool, optional): Whether to unzip the downloaded files. Defaults to True.
        delete (bool, optional): Whether to delete the zip file after unzipping. Defaults to True.
        threads (int, optional): The number of threads to use for parallel downloads. Defaults to 1.
    """

    def _download_one(url, dir, unzip, delete):
        """Helper function to download and process a single URL."""
        f = Path(dir) / Path(url).name  # Filename
        dir = f.parent
        dir.mkdir(parents=True, exist_ok=True)  # Create dir if it does not exist

        print(f"Downloading {url} to {f}...")
        download_url_to_file(url, f, progress=True)

        if unzip and f.suffix == ".zip":
            print(f"Unzipping {f}...")
            with zipfile.ZipFile(f, "r") as zip_ref:
                zip_ref.extractall(path=dir)  # Unzip to dir
            if delete:
                f.unlink()  # Delete zip file

    urls = [url] if isinstance(url, str) else url
    if threads > 1 and len(urls) > 1:
        with ThreadPoolExecutor(threads) as executor:
            executor.map(_download_one, urls, [dir] * len(urls), [unzip] * len(urls), [delete] * len(urls))
    else:
        for u in urls:
            _download_one(u, dir, unzip, delete)

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
