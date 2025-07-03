import os
from ultralytics.utils.downloads import download

# Base directory to save the COCO dataset
base_dir = "E:/datasets/coco_dataset"
os.makedirs(base_dir, exist_ok=True)

# URLs for COCO 2017 dataset (you can modify for other years like 2014)
coco_urls = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    # "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    # Add more URLs if needed (e.g., test images, captions)
    "test_images": "http://images.cocodataset.org/zips/test2017.zip",
    "yolo": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
}


# Main function to download and organize the COCO dataset
def download_coco_dataset():
    urls = list(coco_urls.values())
    download(urls, base_dir, threads=4)
    print("COCO dataset download and extraction complete!")


if __name__ == "__main__":
    try:
        download_coco_dataset()
    except Exception as e:
        print(f"An error occurred: {e}")
