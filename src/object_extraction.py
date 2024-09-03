import torch
from torchvision.ops import masks_to_boxes
import os
from PIL import Image

def extract_objects(predictions, image_path):
    # Check if the image path is valid
    if not os.path.isfile(image_path):
        raise ValueError(f"Invalid image path: {image_path}")
    
    masks = predictions['masks']
    
    # Ensure masks are 2D
    if masks.dim() > 2:
        masks = masks.squeeze(1)  # Ensure the mask is 2D by removing the channel dimension
    
    # Now compute bounding boxes
    boxes = masks_to_boxes(masks)
    
    # Load original image
    image = Image.open(image_path)
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract objects and save
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.int().tolist()  # Convert tensor to a list of integers
        obj_image = image.crop((xmin, ymin, xmax, ymax))
        
        # Save each object with a unique ID
        save_path = os.path.join(output_dir, f'object_{i}.png')
        obj_image.save(save_path)

        # Save metadata (you can extend this as needed)
        with open(os.path.join(output_dir, 'metadata.txt'), 'a') as f:
            f.write(f'Object ID: {i}, Bounding Box: {box.tolist()}\n')

# Example Usage
predictions = {
    'masks': torch.rand(3, 256, 256)  # Example: 3 objects, each with a 256x256 mask
}
extract_objects(predictions, 'data\input_image.jpg')
