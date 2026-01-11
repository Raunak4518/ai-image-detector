import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


MODEL_PATH = 'dual_stream_final.pth'
CLASS_NAMES = ['Human', 'AI']  # Update order based on your training (e.g., 0=Human, 1=AI)

@st.cache_resource
def load_model():
    """
    Loads the model. 
    NOTE: If you saved only the state_dict (weights), you must instantiate 
    the model architecture first.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # OPTION A: If you saved the entire model (torch.save(model, 'model.pth'))
        model = torch.load(MODEL_PATH, map_location=device)
        
        # OPTION B: If you saved only weights (torch.save(model.state_dict(), 'model.pth'))
        # You must uncomment the lines below and replace 'YourModelClass' with your actual class
        # model = YourModelClass() 
        # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- 2. IMAGE PREPROCESSING ---
def transform_image(image):
    """
    Apply the same transformations used during training.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Adjust size to match your model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet stats
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- 3. STREAMLIT UI ---
st.title("🕵️‍♂️ AI vs. Human Image Detector")
st.write("Upload an image to detect if it is Real (Human) or AI-Generated.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Run Inference
    if st.button('Analyze Image'):
        model, device = load_model()
        
        if model:
            with st.spinner('Analyzing...'):
                # Preprocess
                input_tensor = transform_image(image).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    
                    # Calculate probabilities (Softmax or Sigmoid depending on your model)
                    # Assuming Softmax for 2 classes here:
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    # Map to label
                    label = CLASS_NAMES[predicted_class.item()]
                    score = confidence.item() * 100
                
                # Display Results
                if label == 'AI':
                    st.error(f"**Prediction:** {label} Generated Image")
                else:
                    st.success(f"**Prediction:** {label} Image")
                    
                st.info(f"**Confidence:** {score:.2f}%")
