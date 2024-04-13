# OCR service and model for text extraction from an invoice image

## Main sections of the project

### API

Exposes the OCR services throught and http server

### Data

Data used for training and validation

### Model

Different weights saved from the training. `weights_final.pth` is used in this model

### Notebooks

General purpose notebooks for visualization and validation of the model

### Services

Wrappers for the main services used for text extraction

### Src

Main folder where Detectron2 is trained

### Test

Test bench for validation of the Detectron2 model results

## Structure

Following the structure described in https://drivendata.github.io/cookiecutter-data-science/