import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Define dataset class
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define model
def get_model():
    model = models.efficientnet_b0(pretrained=True)
    device = torch.device("cpu")
    model.to(device)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # Assuming 5 classes for DR
    return model

def train_validation_set():
    # Load the original train.csv
    train_df = pd.read_csv('C:/Users/mavul/Documents/known/dr_vision/data/train.csv')  # Replace with your actual path
    # Split the data into training and validation sets (e.g., 80% train, 20% validation)
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    # Save the validation data into a new CSV file
    val_data.to_csv('C:/Users/mavul/Documents/known/dr_vision/data/val_labels.csv', index=False)
    # Optionally, you can save the training data as well (if needed)
    train_data.to_csv('C:/Users/mavul/Documents/known/dr_vision/data/train_labels.csv', index=False)
    print(f"Training data saved to 'data/train_labels.csv' and validation data saved to 'data/val_labels.csv'.")
# Define training function
def train_model(train_df, val_df, img_dir, batch_size=32, epochs=10):
    # Define transformations
    print("working in training1")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("working in training2")
    # Create dataset objects
    train_dataset = RetinopathyDataset(train_df, img_dir, transform)
    val_dataset = RetinopathyDataset(val_df, img_dir, transform)
    print("working in training3")
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("working in training4")
    # Initialize the model
    model = get_model()
    print("working in training4")
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("working in training5")
    # Train the model
    best_accuracy = 0
    print("working in training1",epochs)
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        print("working in training11",epoch)
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("working in training22",epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy}%")

        # Save the model if it performs better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'model/model.pth')  # Save the model in the model folder
            print(f"Model saved with accuracy: {accuracy}%")

    print("Training completed!")
# Add entry point for executing the script
if __name__ == "__main__":
    # Here you can load the CSV for your train/val datasets
    # Example:
    #train_validation_set()
    train_df = pd.read_csv('C:/Users/mavul/Documents/known/dr_vision/data/train_labels.csv')  # Your train data CSV path
    val_df = pd.read_csv('C:/Users/mavul/Documents/known/dr_vision/data/val_labels.csv')  # Your validation data CSV path
    img_dir = 'C:/Users/mavul/Documents/known/dr_vision/data/train'  # Directory where images are stored

    # Call the training function
    train_model(train_df, val_df, img_dir, batch_size=32, epochs=10)
