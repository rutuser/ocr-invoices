import uvicorn
import numpy as np
import cv2
from .extract_invoice import ExtractInvoice
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from rich.traceback import install
from rich.console import Console
from typing import Annotated
from fastapi import Form

install()
console = Console()

app = FastAPI()


@app.post("/extract")
async def extract(threshold: Annotated[float, Form()], file: UploadFile = File(...)):
    contents = await file.read()
    print(f"Received threshold: {threshold}")

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    extracted_data = ExtractInvoice.extract(image, threshold=threshold)

    return JSONResponse(
        {"filename": file.filename, "extracted_data": extracted_data}, status_code=200
    )


def start():
    uvicorn.run("ocr_invoices.src.main:app", host="0.0.0.0", port=8000, reload=True)
