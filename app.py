from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Add this import
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI()

# CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can specify more specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define the architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels (RGB), 32 output channels
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel
        
        self.dropout_conv = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer (adjust dimensions based on image size)
        self.dropout = nn.Dropout(p=0.5)  # Dropout
        self.fc2 = nn.Linear(128, 1)  # Output layer with 1 neuron for binary classification

    def forward(self, x):
        # Apply convolutional layers with activation functions
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        
        # Flatten the tensor
        x = x.view(-1, 64 * 16 * 16)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        # x = torch.sigmoid(x)   not needed as we are using BCEWithLogitsLoss
        return x

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Serve static files (frontend) under /static path
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Serve the index.html file directly
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(image).item()  # Get the single sigmoid output value

    # Determine class based on threshold
    predicted = 1 if output > 0.5 else 0  # 1 = defect, 0 = non-defective
    # # Map the prediction to class name (you can modify this based on your classes)
    class_names = ["defect", "no_defect"]
    return {"class_id": predicted, "class_name": class_names[predicted]}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)