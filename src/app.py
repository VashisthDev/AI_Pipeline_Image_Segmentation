import streamlit as st
from main import main  # This is the main function from main.py

def run_pipeline(image_path):
    main(image_path)  # Pass the image_path to the main function in main.py

def app_main():
    st.title("AI Pipeline for Image Segmentation and Object Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        
        # Run the pipeline
        try:
            run_pipeline(image_path)
            st.success("Image processed successfully!")
            # You could also display results here, such as segmented images or descriptions
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    app_main()
