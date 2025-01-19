from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

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

@app.route("/", methods=["GET"])
def index():
    # Render the HTML page (index.html should be in the templates directory)
    return render_template("index.html")

@app.route("/predict/", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(image).item()  # Get the single sigmoid output value

    # Determine class based on threshold
    predicted = 1 if output > 0.5 else 0  # 1 = defect, 0 = non-defective
    class_names = ["defect", "no_defect"]
    return jsonify({"class_id": predicted, "class_name": class_names[predicted]})

if __name__ == "__main__":
    # port = int(os.getenv("PORT", 8000))  # Default port is 8000
    # app.run(host="0.0.0.0", port=port)
    app.run(debug=True)