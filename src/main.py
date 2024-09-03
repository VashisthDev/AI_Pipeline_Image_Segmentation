from segmentation import segment_image, visualize_segmented_image
from object_extraction import extract_objects
from object_identification import identify_objects
from text_extraction import extract_text
from summarization import summarize_attributes
from data_mapping import map_data, save_mapped_data

def main(image_path):
    # Step 1: Segment the image
    predictions = segment_image(image_path)
    
    # Step 2: Visualize the segmented image
    visualize_segmented_image(image_path, predictions)
    
    # Step 3: Extract objects from the image
    extract_objects(predictions, image_path)
    
    descriptions = []
    texts = []
    summaries = []
    
    # Process each extracted object
    for i in range(len(predictions['masks'])):
        obj_image_path = f'data/extracted_objects/object_{i}.png'
        
        # Step 4: Identify the object
        description = identify_objects(obj_image_path)
        descriptions.append(description)
        
        # Step 5: Extract text from the object
        text = extract_text(obj_image_path)
        texts.append(text)
        
        # Step 6: Summarize the object's attributes
        summary = summarize_attributes(text)
        summaries.append(summary)
    
    # Step 7: Map all extracted data
    mapped_data = map_data(predictions, descriptions, texts, summaries)
    save_mapped_data(mapped_data)

if __name__ == "__main__":
    image_path = 'data/input_image.jpg'  # Example image path
    main(image_path)
