# Script to download and prepare dataset on hyperbolic

LOCAL_DIR="/home/ubuntu/datasets/cocohumanparts"

# # Download the dataset
# python data_tools/yoloformat_download_dataset.py --local_dataset_dir $LOCAL_DIR
# # Create fused hands and legs dataset
# python data_tools/remove_hands_legs.py --datafolder $LOCAL_DIR --format_type yolo

# Download dataset in coco format
python data_tools/cocoformat_download_dataset.py --local_dataset_dir $LOCAL_DIR
# Create fused hands and legs dataset in coco format
python data_tools/remove_hands_legs.py --datafolder $LOCAL_DIR --format_type coco

# Remove backup files
rm -rf $LOCAL_DIR/annotations/*backup.json