from detectron2.structures import BoxMode
from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
import json
from detectron2.data import detection_utils as utils
import random
import os


def build_train_aug():
    augs = [
        T.ResizeShortestEdge(
            short_edge_length=(800, 800), max_size=1333, sample_style="choice"
        ),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
    ]
    return augs


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_train_aug())
        return build_detection_train_loader(cfg, mapper=mapper)


def preprocess_json_for_detectron2(json_file, image_dir):
    """
    Preprocess JSON data for Detectron2, extracting the actual image name.

    Args:
    - json_file: Path to the JSON file containing the annotations.
    - image_dir: Directory where the images are stored.

    Returns:
    - A list of dictionaries in Detectron2's expected format.
    """
    # Load the JSON data
    with open(json_file) as f:
        data = json.load(f)

    # Define the category mapping
    category_mapping = {
        "vendor": 0,
        "date": 1,
        "items": 2,
        "total": 3,
        "id": 4,
        "location": 5,
    }

    # Process the data
    detectron2_data = []
    for item in data:
        record = {}
        # Extract the actual image name
        image_path_components = item["image"].split("/")
        actual_image_name = image_path_components[-1].split("-")[-1]
        record["file_name"] = os.path.join(image_dir, actual_image_name)
        record["image_id"] = item["id"]
        record["height"] = item["label"][0]["original_height"]
        record["width"] = item["label"][0]["original_width"]

        annotations = []
        for label in item["label"]:
            # Convert the bbox format and normalize the coordinates
            bbox = [label["x"], label["y"], label["width"], label["height"]]
            # Convert coordinates from percentages to pixel values if needed
            bbox[0] = (bbox[0] / 100.0) * record["width"]
            bbox[1] = (bbox[1] / 100.0) * record["height"]
            bbox[2] = (bbox[2] / 100.0) * record["width"]
            bbox[3] = (bbox[3] / 100.0) * record["height"]

            annotation = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": category_mapping.get(
                    label["rectanglelabels"][0], -1
                ),  # Default to -1 if label not found
            }
            annotations.append(annotation)

        record["annotations"] = annotations
        detectron2_data.append(record)

    return detectron2_data


def setup_config(num_classes, train_dataset_name, val_dataset_name):
    """
    Set up the Detectron2 model configuration for Faster R-CNN with ResNet-50 FPN.

    Args:
    - num_classes: Number of classes in your dataset (excluding the background class).

    Returns:
    - cfg: The model configuration.
    """
    cfg = get_cfg()
    # Load the Faster R-CNN with ResNet-50 FPN configuration from Detectron2's model zoo
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )

    # Set the number of classes. +1 for the background class.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes + 1

    # Set the path to the weights of the pre-trained model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )

    # Specify the minimum score threshold to consider a detection. Adjust according to your needs.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.DEVICE = "cpu"

    # Specify other configurations like solver, batch size, and number of iterations as needed
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = (
        2500  # Adjust based on the size of your dataset for better results
    )
    cfg.SOLVER.STEPS = (1000, 2000)

    # Set the dataset names for training and validation
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (
        val_dataset_name,
    )  # Use an empty tuple if you do not have a validation set
    cfg.DATALOADER.NUM_WORKERS = 8

    # Specify the output directory where training logs and model checkpoints will be saved
    cfg.OUTPUT_DIR = "output_v4"

    return cfg


def train_model(cfg):
    # Create a trainer using the provided configuration
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Start training
    trainer.train()


def register_dataset(dataset_dicts, dataset_name, classes):
    """
    Registers a dataset in Detectron2's DatasetCatalog and MetadataCatalog.

    Args:
    - dataset_dicts: The dataset dictionaries returned by preprocess_json_for_detectron2.
    - dataset_name: A unique name for the dataset.
    - classes: A list of class names in the dataset.
    """
    # Register the dataset in Detectron2's catalog
    DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)


### TRAINING
json_file = "invoice_labels_v2.json"
image_dir = "labeled_v2"
data_for_detectron2 = preprocess_json_for_detectron2(json_file, image_dir)

# Shuffle the dataset to ensure random distribution
random.shuffle(data_for_detectron2)

# Define the split ratio for training data (e.g., 80% for training)
train_ratio = 0.8
split_index = int(len(data_for_detectron2) * train_ratio)

# Split the dataset
train_dataset = data_for_detectron2[:split_index]
val_dataset = data_for_detectron2[split_index:]

classes = ["vendor", "date", "items", "total", "id", "location"]

# Register train dataset
train_dataset_name = "invoice_dataset_train"
register_dataset(train_dataset, train_dataset_name, classes)

# Register train dataset
val_dataset_name = "invoice_dataset_val"
register_dataset(val_dataset, val_dataset_name, classes)

# Setup model configuration
cfg = setup_config(len(classes), train_dataset_name, val_dataset_name)


if __name__ == "__main__":
    # Start fine-tuning
    train_model(cfg)
