import torch
from PIL import Image
import clip
import os

# Load CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def identify_objects(image_path):
    # Load and preprocess image
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)  # Ensure the image is 4D [1, 3, H, W]
    
    # Encode image using CLIP model
    with torch.no_grad():
        image_features = model.encode_image(image)
    
    # Generate text descriptions
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ["cat", "dog", "car", "tree"]]).to(device)
    
    # Encode text
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # Compute similarity
    logits_per_image = (image_features @ text_features.T).squeeze(0)  # Adjusted to remove extra dimension
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the most likely text description
    index = probs.argmax()
    description = ["cat", "dog", "car", "tree"][index]
    
    return description

# Example Usage
description = identify_objects("data/input_image.jpg")
print(description)
