from img2table.document import Image
from typing import Union
from pathlib import Path
from io import BytesIO
from pandas import DataFrame
from img2table.ocr import TesseractOCR


class TableOCR:
    @staticmethod
    def detect_table(
        src: str | Path | BytesIO | bytes, borderless_tables=True
    ) -> Union[DataFrame, None]:
        print(f"Extracting tables from {src}")
        img = Image(src)
        tesseract = TesseractOCR()

        # Extract tables with Tesseract and PaddleOCR
        tables = img.extract_tables(ocr=tesseract, borderless_tables=borderless_tables)

        if len(tables) > 0:
            return tables[0].df
        else:
            tables = img.extract_tables(
                ocr=tesseract, borderless_tables=not borderless_tables
            )

            if len(tables) > 0:
                return tables[0].df

        return None
