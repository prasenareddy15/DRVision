import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from datetime import datetime
import os

from train import RetinopathyDataset, get_model  # Ensure these are correctly imported
criterion = torch.nn.CrossEntropyLoss()
val_loss=0
# Paths
val_csv_path = 'C:/Users/mavul/Documents/known/DRVision/data/val_labels.csv'
img_dir = 'C:/Users/mavul/Documents/known/DRVision/data/train'
model_path = 'C:/Users/mavul/Documents/known/DRVision/model/model.pth'
results_file = 'C:/Users/mavul/Documents/known/DRVision/model/evaluation_results.csv'

# Load data
val_df = pd.read_csv(val_csv_path)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = RetinopathyDataset(val_df, img_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = get_model()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Metrics
cm = confusion_matrix(all_labels, all_preds)
TP = cm.diagonal()
FN = cm.sum(axis=1) - TP
FP = cm.sum(axis=0) - TP
TN = cm.sum() - (TP + FN + FP)

sensitivity = TP / (TP + FN + 1e-6)
specificity = TN / (TN + FP + 1e-6)
accuracy = (TP.sum() / cm.sum()) * 100

# Prepare row to save
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

row = {
    'timestamp': now ,
    '\naccuracy': accuracy,
    '\nval_loss': val_loss,
    '\navg_sensitivity': np.mean(sensitivity),
    '\navg_specificity': np.mean(specificity),
    '\nsensitivity_per_class': list(sensitivity),
    '\nspecificity_per_class': list(specificity),
}


# Save to CSV
df_row = pd.DataFrame([row])
if os.path.exists(results_file):
    df_row.to_csv(results_file, mode='a', header=False, index=False)
else:
    df_row.to_csv(results_file, index=False)

print(f"Metrics saved to {results_file}")
print(f"Accuracy: {accuracy:.2f}%")
print("Sensitivity per class:", sensitivity)
print("Specificity per class:", specificity)
