from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from ..models.train_model import setup_config
import os


def visualize():
    # Ensure the setup_model_config function sets the cfg.MODEL.WEIGHTS to "output/model_final.pth"
    dataset_name = "invoice_dataset_train"
    cfg = setup_config(
        num_classes=6,
        train_dataset_name=dataset_name,
        val_dataset_name="invoice_dataset_val",
    )
    cfg.MODEL.WEIGHTS = os.path.join(
        "ocr_invoices/src/services/bounds", "weights_final.pth"
    )  # Path to the trained model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the detection threshold

    # Create a DefaultPredictor for inference
    predictor = DefaultPredictor(cfg)

    # Load an image
    image_path = "labeled/val.png"
    image = cv2.imread(image_path)

    # Perform inference
    outputs = predictor(image)
    print(outputs)
    # Visualize the results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=3)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Inference", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


visualize()
