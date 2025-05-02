import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from predict import get_prediction

appl = FastAPI()

# Root route to check if API is up
@appl.get("/")
def read_root():
    return {"message": "DRVision AI Agent"}

# Predict endpoint for image prediction
@appl.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    temp_image_path = os.path.join("temp_images", file.filename)

    # Save the uploaded image temporarily
    os.makedirs("temp_images", exist_ok=True)
    with open(temp_image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Get prediction
    result = get_prediction(temp_image_path)

    # Optionally move the image to the outputs folder after prediction
    output_image_path = os.path.join("outputs", file.filename)
    os.makedirs("outputs", exist_ok=True)
    shutil.move(temp_image_path, output_image_path)

    return JSONResponse(content=result)
