import os
from typing import Dict, TypedDict, Tuple
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from rich.console import Console
from torch import Tensor

console = Console()


class InstanceFields(TypedDict):
    pred_boxes: Boxes
    scores: Tensor
    pred_classes: Tensor


class BoundsRecognition:
    def setup_config(num_classes, dataset_name):
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
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )

        # Set the number of classes. +1 for the background class.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes + 1

        # Set the path to the weights of the pre-trained model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        )

        # Specify the minimum score threshold to consider a detection. Adjust according to your needs.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg.MODEL.DEVICE = "cpu"
        cfg.TEST.EVAL_PERIOD = 20

        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[2.0, 3.0, 4.0, 5.0]]

        # Specify other configurations like solver, batch size, and number of iterations as needed
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = (
            2500  # Adjust based on the size of your dataset for better results
        )
        cfg.SOLVER.STEPS = (1500, 2000)  # At which point to change the LR
        cfg.SOLVER.GAMMA = 0.1  # LR decay factor

        # Set the dataset names for training and validation
        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = 8

        return cfg

    def add_margin_to_box(
        bounding_box: Tuple[int, int, int, int],
        marginX: int,
        marginY: int,
        image_shape: Tuple[int, int],
    ):
        """
        Add a margin to bounding boxes to ensure text is not cropped while keeping the boxes within image boundaries.

        :param bounding_boxes: List of bounding box coordinates in the format (startX, startY, endX, endY).
        :param marginX: Number of pixels to add as a margin to the left and right sides of each box.
        :param marginY: Number of pixels to add as a margin to the top and bottom sides of each box.
        :param image_shape: The shape of the image given as (height, width), used to ensure boxes stay within bounds.
        :return: List of adjusted bounding boxes with margins applied without exceeding the image boundaries.
        """
        startX, startY, endX, endY = bounding_box
        # Calculate the new coordinates with margin, ensuring they are within the image bounds
        new_startX = max(0, startX - marginX)
        new_startY = max(0, startY - marginY)
        new_endX = min(
            image_shape[1] - 1, endX + marginX
        )  # Ensure not to exceed image width
        new_endY = min(image_shape[0] - 1, endY + marginY)

        return (new_startX, new_startY, new_endX, new_endY)

    def get_recognition(image, threshold: float):
        # Ensure the setup_model_config function sets the cfg.MODEL.WEIGHTS to "output/model_final.pth"
        dataset_name = "invoice_dataset_train"
        cfg = BoundsRecognition.setup_config(6, dataset_name)
        cfg.MODEL.WEIGHTS = os.path.join(
            "ocr_invoices/src/services/bounds", "weights_final.pth"
        )  # Path to the trained model weights
        print(f"RECEIVED THRESHOLD: {threshold}")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Set the detection threshold

        # Create a DefaultPredictor for inference
        predictor = DefaultPredictor(cfg)

        # Perform inference
        outputs: Dict[str, Instances] = predictor(image)

        return outputs
