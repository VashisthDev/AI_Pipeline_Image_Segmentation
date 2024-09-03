import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

def segment_image(image_path):
    # Load the pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # Load the image and convert it to a tensor
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    
    # Perform inference
    with torch.no_grad():
        predictions = model([image_tensor])
    
    return predictions[0]  # Return the first (and only) prediction

def visualize_segmented_image(image_path, predictions, threshold=0.5):
    import matplotlib.pyplot as plt
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw the masks
    for i, mask in enumerate(predictions['masks']):
        if predictions['scores'][i] > threshold:
            mask = mask[0].mul(255).byte
