import mimetypes
import requests
import pandas as pd
import json
from .utils.match_percentage import match_percentage, mean_match_percentage


def test(threshold=0.5):
    # Load predictions from the JSON file
    with open("test.json", "r") as file:
        predictions = json.load(file)

    # Initialize an empty DataFrame for the test results
    columns = [
        "image",
        "vendor",
        "total",
        "items",
        "date",
        "id",
    ]
    test_results = pd.DataFrame(columns=columns)
    text_results = pd.DataFrame(columns=columns)

    # Define the URL of the OCR system
    url = "http://localhost:8000/extract"

    # Iterate over each image in the predictions
    for prediction in predictions:
        image_path = f"test/{prediction['image']}"  # Adjust the path as necessary
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "application/octet-stream"  # Fallback MIME type

        # Open the image file and keep it open for the request
        with open(image_path, "rb") as img_file:
            files = [("file", (prediction["image"], img_file, mime_type))]
            headers = {}

            # Send the request to the OCR system
            response = requests.post(
                url, headers=headers, files=files, data={"threshold": threshold}
            )
            if response.status_code == 200:
                ocr_result = response.json()["extracted_data"]

                # Compare OCR results with predictions
                vendor_match = match_percentage(
                    prediction["vendor"], ocr_result["vendor"]
                )
                total_match = match_percentage(prediction["total"], ocr_result["total"])
                items_match = 0
                if ocr_result["items"]:  # Ensure there are OCR items to compare
                    items_match = mean_match_percentage(
                        prediction["items"], ocr_result["items"][0]
                    )
                date_match = match_percentage(prediction["date"], ocr_result["date"])
                id_match = match_percentage(prediction["id"], ocr_result["id"])

                # Append the comparison results to the DataFrame
                row = pd.DataFrame(
                    [
                        {
                            "image": prediction["image"],
                            "vendor": vendor_match,
                            "total": total_match,
                            "items": items_match,
                            "date": date_match,
                            "id": id_match,
                        },
                    ]
                )

                text_row = pd.DataFrame(
                    [
                        {
                            "image": prediction["image"],
                            "vendor": ocr_result["vendor"],
                            "total": ocr_result["total"],
                            "items": ocr_result["items"],
                            "date": ocr_result["date"],
                            "id": ocr_result["id"],
                        },
                    ]
                )

                test_results = pd.concat([test_results, row], ignore_index=True)
                text_results = pd.concat([text_results, text_row], ignore_index=True)
            else:
                print(f"Failed to process {prediction['image']}")
                print(f"Error in response: {response.text}")

    # Close all open files
    for prediction in predictions:
        image_path = f"/test/{prediction['image']}"  # Adjust the path as necessary
        files[0][1][1].close()

    # Optionally, save the DataFrame to a CSV file for further analysis
    test_results.to_csv(f"ocr_test_results-aug-v4-{threshold}v2.csv", index=False)
    text_results.to_csv(f"ocr_test_results-aug-v4-text-{threshold}v2.csv", index=False)


# For testing with other thresholds uncomment the following lines
# print("Testing with different thresholds:")
# for i in [0.5, 0.4, 0.3, 0.2]:
#     print(f"Threshold: {i}")
#     test(i)

# Test with the default threshold of 0.5
test(0.5)
