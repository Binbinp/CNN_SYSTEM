import fitz  # PyMuPDF for PDF processing 
import numpy as np
import os
from PIL import Image

# Path configurations
input_folder = r'C:\\Users\\acer\\OneDrive\\Documents\\Input_Training_Data'  # PDF folder path
output_folder = r'C:\\Users\\acer\\OneDrive\\Documents\\Output_Training_Data'  # Output folder path
image_size = (1024, 1024)  # Increase target size for CNN input

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

def preprocess_pdf(pdf_path, output_folder):
    pdf_file = fitz.open(pdf_path)
    for page_num in range(len(pdf_file)):
        page = pdf_file.load_page(page_num)
        # Extract the pixmap with a higher resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # Adjust the matrix for higher resolution
        
        # Convert to PIL image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize the image using LANCZOS
        img = img.resize(image_size, Image.LANCZOS)
        
        # Convert to numpy array for further processing
        img_np = np.array(img)
        
        # Normalize the image if needed
        img_np = img_np / 255.0
        
        # Convert back to PIL Image if needed
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # Save the preprocessed image
        image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num + 1}.png"
        img_pil.save(os.path.join(output_folder, image_filename))

def preprocess_pdfs_in_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            preprocess_pdf(pdf_path, output_folder)
            print(f"Processed {filename}")

# Run preprocessing
preprocess_pdfs_in_folder(input_folder, output_folder)
