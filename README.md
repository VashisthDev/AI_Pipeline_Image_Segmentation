# AI Pipeline for Image Segmentation and Object Analysis

This project implements a robust AI pipeline that performs image segmentation, object identification, text extraction, and summarization of object attributes. The entire pipeline is modularized to ensure flexibility, scalability, and ease of integration.

## Features

- **Image Segmentation**: Automatically segments objects within an image using a deep learning model.
- **Object Identification**: Recognizes and labels segmented objects.
- **Text Extraction**: Extracts text embedded within objects using OCR.
- **Attribute Summarization**: Summarizes key attributes of objects for easier analysis.
- **Streamlit UI**: A user-friendly interface to upload images, visualize segmentation, and review analysis results.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VashisthDev/AI_Pipeline_Image_Segmentation.git
   cd AI_Pipeline_Image_Segmentation

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3.Run the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py 
   ```
## Usage

Upload Image: Use the Streamlit interface to upload an image.
View Segmentation: The app will display segmented objects overlaid on the original image.
Object Details: Review the extracted object images, descriptions, texts, and summarized attributes.
Final Output: View the annotated image and a table containing all mapped data for each object.

