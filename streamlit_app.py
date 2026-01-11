import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- 1. CONFIGURATION ---
# Make sure this matches the name of the file you upload to GitHub
MODEL_PATH = 'dual_stream_final.pth' 

# --- 2. DEFINE THE MODEL ARCHITECTURE ---
# We paste the exact class from your training code here.
# NOTE: We set weights=None to avoid downloading ImageNet weights, 
# since we are about to overwrite them with your trained weights anyway.
class DualStreamNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Stream 1 (EfficientNet B4)
        self.stream1 = models.efficientnet_b4(weights=None)
        self.stream1.classifier = nn.Identity()

        # Stream 2 (DenseNet 121)
        self.stream2 = models.densenet121(weights=None)
        self.stream2.classifier = nn.Identity()

        # Fusion head
        self.fc = nn.Sequential(
            nn.Linear(1792 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        f1 = self.stream1(x)
        f2 = self.stream2(x)
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        device = torch.device('cpu') # Force CPU for Streamlit Cloud
        model = DualStreamNet()
        
        # Load the weights
        # map_location='cpu' is crucial for deploying on servers without GPUs
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model. Make sure '{MODEL_PATH}' is in the same folder.")
        st.error(f"Details: {e}")
        return None

# --- 4. PREPROCESSING ---
# Using the exact same transforms as your validation set
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- 5. APP INTERFACE ---
st.title("🛡️ AI Image Detector")
st.write("Upload an image to check if it's **Real** or **AI-Generated**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image'):
        with st.spinner('Scanning...'):
            model = load_model()
            
            if model:
                # Prepare input
                input_tensor = process_image(image)
                
                # Inference
                with torch.no_grad():
                    logit = model(input_tensor)
                    probability = torch.sigmoid(logit).item()
                
                # Logic: In your training, 1_fake means 1 is Fake.
                # Probability > 0.5 implies Class 1 (Fake)
                is_fake = probability > 0.5
                
                # Dynamic Color & Label
                if is_fake:
                    confidence = probability
                    st.error(f"### 🚨 Prediction: AI-GENERATED (FAKE)")
                    st.progress(confidence)
                    st.write(f"**Confidence:** {confidence:.2%}")
                else:
                    confidence = 1 - probability
                    st.success(f"### ✅ Prediction: REAL HUMAN IMAGE")
                    st.progress(confidence)
                    st.write(f"**Confidence:** {confidence:.2%}")
