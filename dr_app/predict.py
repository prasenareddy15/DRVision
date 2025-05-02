import torch
from torchvision import transforms
from PIL import Image
import os
from model.train import get_model

# Define the same transformation used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model and weights
model = get_model()
model.load_state_dict(torch.load("C:/Users/mavul/Documents/known/dr_vision/model/model.pth", map_location=torch.device("cpu")))
model.eval()

# Predict function
def get_prediction(image_path):
    if not os.path.exists(image_path):
        return {"error": "Image not found."}
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return {"prediction": prediction}
