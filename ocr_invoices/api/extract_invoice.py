import datetime
from io import BytesIO
from typing import Dict, List, Tuple, TypedDict

import cv2
import pytesseract
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from rich.console import Console

from ..services.boundsRecognition import BoundsRecognition
from ..services.tableExtraction import TableOCR
from .utils.parsers import Parsers

DEBUG = True

console = Console()

BoundingBoxes = Dict[str, Tuple[int, int, int, int] | None]


class PredictedClasses(TypedDict):
    vendor: str | None
    total: str | None
    id: str | None
    date: str | None
    location: str | None
    items: List[List[str]] | None


class ResultClasses(TypedDict):
    vendor: str | None
    total: str | None
    id: str | None
    date: datetime.date | None
    location: str | None
    items: List[List[str]] | None


class ExtractInvoice:
    """`Main class for extracing information from the invoice image using OCR`"""

    @staticmethod
    def __preprocess_image(image: MatLike) -> MatLike:
        """
        `Preprocess the image before extracting text`

        Args:
         - image: The image to be preprocessed

        Returns:
        - The preprocessed image
        """
        console.print("[PREPROCESSING] processing image")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Thresholding
        processed_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Debug: Show the preprocessed image
        plt.figure(figsize=(10, 10))
        plt.imshow(processed_image, cmap="gray")
        plt.title("Preprocessed Image")
        plt.show()

        return processed_image

    @staticmethod
    def __detect_text_boundings(
        preprocessed_image: MatLike,
        add_margin: bool,
        threshold: float,
    ) -> BoundingBoxes:
        """
        `Detect text bounding boxes using Detectron2`

        Args:
            - preprocessed_image: The preprocessed image to detect text bounding boxes
            - add_margin: Whether to add margin to the bounding boxes
            - threshold: The threshold for the detection

        Returns:
            - The bounding boxes of the detected text

        """
        console.log(
            "[DETECTING TEXT BOUNDINGS] start detecting bounding boxes using Detectron2"
        )

        # Save the preprocessed image to a file if not already a file path
        if DEBUG:
            image_path = "path_to_preprocessed_image.jpg"
            cv2.imwrite(image_path, preprocessed_image)

        # Use Detectron2 to get detections
        detections = BoundsRecognition.get_recognition(preprocessed_image, threshold)

        # Mapping of category labels to their corresponding class indices
        category_mapping = {
            "vendor": 0,
            "date": 1,
            "items": 2,
            "total": 3,
            "id": 4,
            "location": 5,
        }

        # Reverse the mapping for convenience
        index_to_category = {v: k for k, v in category_mapping.items()}

        scores = detections["instances"].scores.numpy()
        class_labels = detections["instances"].pred_classes.numpy()
        boxes = detections["instances"].pred_boxes.tensor.numpy()

        # Initialize with None values
        highest_score_bounding_boxes = {category: None for category in category_mapping}
        # Initialize with 0 scores
        highest_scores = {category: 0 for category in category_mapping}

        # Tracking the highest score bounding boxes per category
        for box, score, label in zip(boxes, scores, class_labels):
            category = index_to_category.get(label)
            if category and score > highest_scores[category]:
                bounds = (
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                )

                if add_margin:
                    highest_score_bounding_boxes[category] = (
                        BoundsRecognition.add_margin_to_box(
                            bounds, 10, 10, preprocessed_image.shape
                        )
                    )
                else:
                    highest_score_bounding_boxes[category] = bounds

                highest_scores[category] = score

        console.log(
            "[DETECTING TEXT BOUNDINGS] selected bounding boxes with highest scores for each category"
        )
        return highest_score_bounding_boxes

    @staticmethod
    def __extract_text(
        image: MatLike,
        bounding_boxes: BoundingBoxes,
        psm="--psm 7",
    ) -> PredictedClasses:
        """
        `Extract text from the image using the bounding boxes`

        Args:
            - image: The image to extract text from
            - bounding_boxes: The bounding boxes of the image sections
            - psm: The Page Segmentation Mode for Tesseract

        Returns:
            - The extracted text from the image

        """
        console.log("[EXTRACTION] extracting text from image")

        extracted_texts: PredictedClasses = {
            "total": None,
            "items": None,
            "vendor": None,
            "date": None,
            "id": None,
            "location": None,
        }

        for category, box in bounding_boxes.items():
            if not box:
                continue

            startX, startY, endX, endY = box
            roi = image[startY:endY, startX:endX]

            # Save the ROI to a file
            if DEBUG:
                cv2.imwrite(f"./ROI-{category}" + ".jpg", roi)

            if category == "date":
                text = pytesseract.image_to_string(roi, config="--psm 6")
                extracted_texts["date"] = text
            elif category == "items":
                is_success, buffer = cv2.imencode(".jpg", roi)
                if is_success:
                    bytes_io = BytesIO(buffer)
                    df = TableOCR.detect_table(bytes_io, True)

                    extracted_texts["items"] = (
                        df.to_numpy().tolist() if df is not None else None
                    )
            else:
                text = pytesseract.image_to_string(roi, config=psm)
                extracted_texts[category] = text

        return extracted_texts

    @staticmethod
    def __clean_predictions(predictions: PredictedClasses) -> ResultClasses:
        """
        `Clean the extracted predictions`

        Args:
            - predictions: The extracted predictions

        Returns:
            - The cleaned predictions
        """
        vendor = (
            Parsers.parse_general(predictions["vendor"])
            if predictions["vendor"]
            else None
        )
        total = (
            Parsers.parse_totals(predictions["total"]) if predictions["total"] else None
        )
        id = Parsers.parse_general(predictions["id"]) if predictions["id"] else None
        date = Parsers.extract_date(predictions["date"])
        location = (
            Parsers.parse_general(predictions["location"])
            if predictions["location"]
            else None
        )
        items = predictions["items"]

        return ResultClasses(vendor, total, id, date, location, items)

    @staticmethod
    def extract(image: MatLike, threshold: float = 0.5, psm="--psm 7") -> ResultClasses:
        """
        `Extract information from the invoice image`

        Args:
            - image: The invoice image
            - threshold: The threshold for the detection
            - psm: The Page Segmentation Mode for Tesseract

        Returns:
            - The extracted information from the invoice image
        """
        # Preprocess the image
        preprocessed_image: MatLike = ExtractInvoice.__preprocess_image(
            image_path=image
        )

        # Detect text bounding boxes using the preprocessed image
        bounding_boxes = ExtractInvoice.__detect_text_boundings(
            preprocessed_image=preprocessed_image, add_margin=False, threshold=threshold
        )

        # Extract text using the adjusted bounding boxes
        extracted_boudings_texts = ExtractInvoice.__extract_text(
            image, bounding_boxes, psm
        )

        # Clean the extracted predictions
        cleaned_extracted_boudings_texts = ExtractInvoice.__clean_predictions(
            extracted_boudings_texts
        )

        return cleaned_extracted_boudings_texts
