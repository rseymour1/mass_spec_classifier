#!pip install depthcharge-ms
#!pip install pylance==0.15

import depthcharge as dc
from depthcharge.encoders import PeakEncoder
from depthcharge.data import StreamingSpectrumDataset
#import torch

mzml = './orbitrap_lumos/08CPTAC_C_GBM_W_PNNL_20210830_B2S3_f20.mzML'
processing_fn = [
    #https://spectrum-utils.readthedocs.io/en/latest/api.html#spectrum_utils.spectrum.MsmsSpectrum.set_mz_range
    dc.data.preprocessing.set_mz_range(min_mz=0),
    #https://spectrum-utils.readthedocs.io/en/latest/api.html#spectrum_utils.spectrum.MsmsSpectrum.filter_intensity
    dc.data.preprocessing.filter_intensity(min_intensity = 0.1),  # Change this value to allow number of peaks per spectrum
    #https://spectrum-utils.readthedocs.io/en/latest/api.html#spectrum_utils.spectrum.MsmsSpectrum.scale_intensity
    dc.data.preprocessing.scale_intensity(scaling=None),  # Might want to change later
    dc.data.preprocessing.scale_to_unit_norm,  # Not sure what this is yet
]
df = dc.data.spectra_to_df(
    mzml,
    progress=True,
    preprocessing_fn=processing_fn
)

encoder = PeakEncoder(100)
dataset = StreamingSpectrumDataset(df, batch_size=2)

from depthcharge.transformers import SpectrumTransformerEncoder

d_model = 128
model = SpectrumTransformerEncoder(d_model=d_model, 
                                   nhead=8, 
                                   dim_feedforward=1024, 
                                   n_layers=1, 
                                   dropout=0.0, 
                                   peak_encoder= PeakEncoder(d_model=d_model, 
                                                             min_mz_wavelength=0.001, 
                                                             max_mz_wavelength=10000, 
                                                             min_intensity_wavelength=1e-06, 
                                                             max_intensity_wavelength=1, 
                                                             learnable_wavelengths=False) )
print("Loaded Model")
for batch in dataset:
    out = model(batch["mz_array"], batch["intensity_array"])
    print(out[0].shape)
    break

###### Code from perplexity #####
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Define the classifier
class MassSpecClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.d_model, num_classes)
    
    def forward(self, mz, intensity):
        encoded = self.encoder(mz, intensity)
        return self.classifier(encoded[:, 0, :])  # Use the [CLS] token for classification

# Prepare the dataset
class MassSpecDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_column):
        self.df = df
        self.label_column = label_column
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "mz_array": torch.tensor(row["mz_array"]),
            "intensity_array": torch.tensor(row["intensity_array"]),
            "label": torch.tensor(row[self.label_column])
        }

# Assuming you have a label column in your DataFrame
label_column = "mass_spectrometer"
num_classes = df[label_column].nunique()

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = MassSpecDataset(train_df, label_column)
val_dataset = MassSpecDataset(val_df, label_column)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
classifier = MassSpecClassifier(model, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

for epoch in range(num_epochs):
    classifier.train()
    for batch in train_loader:
        mz = batch["mz_array"].to(device)
        intensity = batch["intensity_array"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = classifier(mz, intensity)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            mz = batch["mz_array"].to(device)
            intensity = batch["intensity_array"].to(device)
            labels = batch["label"].to(device)
            
            outputs = classifier(mz, intensity)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")

# Save the model
torch.save(classifier.state_dict(), "mass_spec_classifier.pth")
'''