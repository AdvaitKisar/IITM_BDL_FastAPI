#---------------------------------- MAIN FILE  -------------------------------
# Importing libraries
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn, io

# Importing support functions from python scripts
from load_model import load_model
from predict_digit import predict_digit
from format_image import format_image

# Creating App
app = FastAPI(title="AE20B007: Digit Recognition App for Big Data Lab")
path = 'MNIST_Model.keras' # Specifying path of model

#---------------------------------- TASK 1 ------------------------------------
# First API Endpoint for predicting digits in the MNIST Images
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    '''
    This is Task 1

    Function: API Endpoint which takes the uploaded input image of size 28X28 and predicts the digits

    Input:-
    file [File]: File uploaded by user

    Output:-
    [dict]: Key is "digit" and value is the digit predicted by the model
    '''
    content = await file.read() # Content of uploaded file is read
    image = Image.open(io.BytesIO(content)) # Image is opened as an PIL Image object

    img = np.array(image) # Converted to numpy array
    img = img.reshape(1, -1) # Array is flattened to 1d array with 784 elements

    path = 'MNIST_Model.keras' # Path of the model
    model = load_model(path) # Model is loaded

    digit = predict_digit(model, img) # Digit is predicted
    return {"digit": digit} # Output is returned

#---------------------------------- TASK 2 ------------------------------------
# Second API Endpoint for predicting digits from images of any size
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    '''
    This is Task 2

    Function: API Endpoint which takes the uploaded input image of any size and predicts the digits

    Input:-
    file [File]: File uploaded by user

    Output:-
    [dict]: Key is "digit" and value is the digit predicted by the model
    '''
    content = await file.read() # Content of uploaded file is read
    image = Image.open(io.BytesIO(content)) # Image is opened as an PIL Image object
    img = format_image(image) # Image is resized to 28X28 grayscale image

    path = 'MNIST_Model.keras' # Path of the model
    model = load_model(path) # Model is loaded

    digit = predict_digit(model, img) # Digit is predicted
    return {"digit": digit} # Output is returned

# This app is now hosted at http://0.0.0.0:7000
uvicorn.run(app, host='0.0.0.0', port=7000)