def summarize_attributes(text):
    # Simple summary logic for demonstration
    if len(text.strip()) == 0:
        summary = "No text found"
    else:
        summary = f"Extracted text: {text[:30]}..."
    
    return summary
