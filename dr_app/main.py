import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dr_app.predict import get_prediction

apple = FastAPI()
print("cable")
# Root route to check if API is up
@apple.get("/")
def read_root():
    print("cable1")
    return {"message": "DRVision AI Agent"}
print("cable2")
# Predict endpoint for image prediction
@apple.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    temp_image_path = os.path.join("temp_images", file.filename)
    print("cable3")
    # Save the uploaded image temporarily
    os.makedirs("temp_images", exist_ok=True)
    with open(temp_image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print("cable4")
    # Get prediction
    result = get_prediction(temp_image_path)

    # Optionally move the image to the outputs folder after prediction
    output_image_path = os.path.join("outputs", file.filename)
    os.makedirs("outputs", exist_ok=True)
    shutil.move(temp_image_path, output_image_path)
    print("5")
    return JSONResponse(content=result)
