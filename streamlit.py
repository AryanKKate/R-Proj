import streamlit as st
from PIL import Image
import torch
import cohere
from transformers import BlipProcessor, BlipForConditionalGeneration

# Remove or comment out these lines
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
co = cohere.Client("7HpPc5ghLeUjTWUHljnX8Y7xgxRcRhdOUg2bv9Px")

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# Remove this function since we're not using OCR anymore
# def extract_text_from_image(image):
#     return pytesseract.image_to_string(image)

def query_cohere(caption, user_question):
    prompt = f"Image Caption: {caption}\nUser Question: {user_question}\nAnswer:"
    
    response = co.generate(
        model="command-xlarge", prompt=prompt, max_tokens=100, temperature=0.75
    )
    return response.generations[0].text.strip()

# Streamlit UI
st.title("VisioNiX - Image Captioning and Q&A")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    #st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    user_question = st.text_input("Ask a question about the image")
    
    if st.button("Get Answer") and user_question:
        caption = generate_caption(image)
        answer = query_cohere(caption, user_question)
        
        st.write("### Answer:", answer)
