import os
import numpy as np
from datetime import datetime
import pytesseract  # Tesseract OCR
from PIL import Image, ImageTk
import fitz  # PyMuPDF for handling PDFs
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, messagebox
import pandas as pd  # For handling Excel files
import re


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows

# Load your pre-trained model
model = load_model('model\cnnmodelL2Dropout_16BS.h5')  # Make sure to replace with your actual model path
class_labels = ['with_both', 'with_image_only', 'with_no_data', 'with_signature_only', 'with_wrong_image', 'other_class']  # Example class labels

# Variable to store the last extracted text globally
last_extracted_text = ""

# Define required fields for validation
REQUIRED_FIELDS = ["Name", "Date"]

# Function to validate extracted text for missing or illegible fields
def validate_extracted_text(text):
    # Check for missing required fields
    missing_fields = [field for field in REQUIRED_FIELDS if field not in text]
    
    # Log results
    if missing_fields:
        return f"Missing fields: {', '.join(missing_fields)}"
    else:
        return "All required fields are present."

""" # Function to extract field content from the text
def extract_field_content(text, field):
    # Define patterns for each field. The field names should match how they're represented in the text.
    patterns = {
        "Full Name": r"Name:\s*(.*)",  # Matches "Name: " followed by any content
        "Date": r"Date:\s*(\S+)",  # Matches "Date: " followed by a non-whitespace string (e.g., a date)
        #"Signature": r"Signature:\s*(.*)"  # Matches "Signature: " followed by content
    }

    # Use the regex pattern for the specific field to extract content
    if field in patterns:
        match = re.search(patterns[field], text)
        if match:
            return match.group(1)  # Return the captured content (group 1)
    return ""  # Return an empty string if no match is found """

# Function to classify a single image
def classify_image(image_path):
    global last_extracted_text
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(384, 384))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # Normalize pixel values

        # Make prediction
        prediction = model.predict(x, verbose=0)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        # OCR text extraction
        last_extracted_text = extract_text_from_image(image_path)

        # Validate extracted text for missing fields
        validation_result = validate_extracted_text(last_extracted_text)

        # Determine acceptance
        if predicted_label == 'with_both' and validation_result == "All required fields are present.":
            return "Accepted", predicted_label
        else:
            return "Rejected", f"{predicted_label}, {validation_result}"
    except Exception as e:
        return "Error", str(e)

# Function to extract text from image using OCR
def extract_text_from_image(image_path):
    try:
        # Open image
        img = Image.open(image_path)
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return str(e)

# Function to show the extracted text in a new window
def show_extracted_text():
    if last_extracted_text:
        text_window = ctk.CTkToplevel()
        text_window.title("Extracted Text")
        
        # Add text widget to display the extracted text
        text_label = ctk.CTkLabel(text_window, text=f"OCR Extracted Text:\n{last_extracted_text}", font=("Arial", 12), wraplength=500)
        text_label.pack(padx=20, pady=20)
    else:
        messagebox.showinfo("No Text", "No text extracted yet.")

def display_rejection_comparison(original_image_path, rejected_image_path, reason):
    original_img = Image.open('image/original.png')  # Correct the path if needed
    rejected_img = Image.open(rejected_image_path)

    original_img = original_img.resize((400, 400))
    rejected_img = rejected_img.resize((400, 400))

    comparison_window = ctk.CTkToplevel()
    comparison_window.title("Rejection Comparison")
    comparison_window.protocol("WM_DELETE_WINDOW", comparison_window.destroy)  # Ensure proper closure

    original_image = ImageTk.PhotoImage(original_img)
    rejected_image = ImageTk.PhotoImage(rejected_img)

    original_label = ctk.CTkLabel(comparison_window, image=original_image, text="Original Image", font=("Arial", 16))
    original_label.image = original_image
    original_label.grid(row=0, column=0, padx=10, pady=10)

    rejected_label = ctk.CTkLabel(comparison_window, image=rejected_image, text="Rejected Image", font=("Arial", 16))
    rejected_label.image = rejected_image
    rejected_label.grid(row=0, column=1, padx=10, pady=10)

    reason_label = ctk.CTkLabel(comparison_window, text=f"Reason for rejection: {reason}", font=("Arial", 16))
    reason_label.grid(row=1, column=0, columnspan=2, pady=10)

    # Add a close button to allow manual closure of the window
    close_button = ctk.CTkButton(comparison_window, text="Close", command=comparison_window.destroy)
    close_button.grid(row=2, column=0, columnspan=2, pady=10)
    
def display_accepted_comparison(original_image_path, accepted_image_path, reason):
    original_img = Image.open('image/original.png')  # Correct the path if needed
    accepted_img = Image.open(accepted_image_path)

    original_img = original_img.resize((400, 400))
    accepted_img = accepted_img.resize((400, 400))

    comparison_window = ctk.CTkToplevel()
    comparison_window.title("Accepted Comparison")
    comparison_window.protocol("WM_DELETE_WINDOW", comparison_window.destroy)  # Ensure proper closure

    original_image = ImageTk.PhotoImage(original_img)
    accepted_image = ImageTk.PhotoImage(accepted_img)

    original_label = ctk.CTkLabel(comparison_window, image=original_image, text="Original Image", font=("Arial", 16))
    original_label.image = original_image
    original_label.grid(row=0, column=0, padx=10, pady=10)

    accepted_label = ctk.CTkLabel(comparison_window, image=accepted_image, text="Accepted Image", font=("Arial", 16))
    accepted_label.image = accepted_image
    accepted_label.grid(row=0, column=1, padx=10, pady=10)

    reason_label = ctk.CTkLabel(comparison_window, text=f"Reason for Accepted: {reason}", font=("Arial", 16))
    reason_label.grid(row=1, column=0, columnspan=2, pady=10)

    # Add a close button to allow manual closure of the window
    close_button = ctk.CTkButton(comparison_window, text="Close", command=comparison_window.destroy)
    close_button.grid(row=2, column=0, columnspan=2, pady=10)
    
def process_single_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("PDF Files", "*.pdf")]
    )
    if not file_path:
        return

    if file_path.lower().endswith('.pdf'):
        pdf_images = convert_pdf_to_images(file_path)
        if not pdf_images:
            messagebox.showerror("Error", "Failed to process PDF.")
            return

        for page_num, image_path in enumerate(pdf_images, start=1):
            result, label = classify_image(image_path)
            if result == "Rejected":
                reason = "Missing signature or image"
                display_rejection_comparison(image_path, image_path, reason)
            elif result == "Accepted":
                reason = "Valid Image: Meets all requirements"
                display_accepted_comparison(image_path, image_path, reason)
            messagebox.showinfo("Result", f"Page {page_num}: {result} ({label})")
            os.remove(image_path)  # Clean up temporary image files
    else:
        result, label = classify_image(file_path)
        if result == "Rejected":
            reason = "Missing signature or image/Incorrect format"
            display_rejection_comparison(file_path, file_path, reason)
        elif result == "Accepted":
            reason = "Meets all requirements"
            display_accepted_comparison(file_path, file_path, reason)
        messagebox.showinfo("Result", f"File: {result} ({label})")


def save_rejection_summary_to_excel(rejection_list):
    if rejection_list:
        # Create a DataFrame from the rejection list
        df = pd.DataFrame(rejection_list, columns=["File Name", "Page Number", "Reason"])

        # Get the current date to use in the default file name
        current_date = datetime.now().strftime("%Y-%m-%d")
        default_file_name = f"rejected_files_summary_{current_date}.xlsx"

        # Ask for a location to save the Excel file, with a default file name
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                                 initialfile=default_file_name,
                                                 filetypes=[("Excel Files", "*.xlsx")])
        if save_path:
            try:
                # Save to Excel
                df.to_excel(save_path, index=False, engine='openpyxl')
                messagebox.showinfo("Success", "Rejection summary saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save rejection summary: {e}")
    else:
        messagebox.showinfo("No Rejections", "No rejections to save.")

# Function to process a folder of images
def process_folder():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return

    results = []
    rejection_list = []
    rejections = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Process PDF files
        if file_path.lower().endswith('.pdf'):
            pdf_images = convert_pdf_to_images(file_path)
            if not pdf_images:
                results.append(f"{file_name}: Failed to process PDF.")
                continue

            for page_num, image_path in enumerate(pdf_images, start=1):
                result, label = classify_image(image_path)
                if result == "Rejected":
                    reason = "Missing signature or image"
                    rejection_list.append([file_name, page_num, reason])
                    rejections.append((image_path, image_path, reason))
                results.append(f"{file_name} (Page {page_num}): {result} ({label})")
                os.remove(image_path)  # Clean up temporary image files

        # Process image files
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            result, label = classify_image(file_path)
            if result == "Rejected":
                reason = "Missing signature or image"
                rejection_list.append([file_name, None, reason])  # For images without pages
                rejections.append((file_path, file_path, reason))
            results.append(f"{file_name}: {result} ({label})")

    # Display rejection summary if any rejections occurred
    if rejection_list:
        save_rejection_summary_to_excel(rejection_list)

    # Display results in a message box
    if results:
        result_text = "\n".join(results)
        messagebox.showinfo("Results", f"Processed Files:\n\n{result_text}")
    else:
        messagebox.showinfo("Results", "No valid files were found in the selected folder.")

# Function to convert a PDF to images
def convert_pdf_to_images(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        image_paths = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            image_path = f"image_{page_num}.png"
            pix.save(image_path)
            image_paths.append(image_path)

        return image_paths
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert PDF: {str(e)}")
        return []

# Function to initialize the GUI
def init_gui():
    global root
    root = TkinterDnD.Tk()
    root.title("Application Form Verification System")
    root.geometry("600x400")
    
    # Set appearance mode and default color theme
    ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"
    
    # Configure the grid to use the whole screen
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=0)  # Sidebar will have fixed width
    root.grid_columnconfigure(1, weight=1)  # Main content will fill available space
    
    # Create the sidebar frame with black background
    sidebar_frame = ctk.CTkFrame(root, fg_color="#0077B6", width=200)
    sidebar_frame.grid(row=0, column=0, sticky="ns")

    # Create appearance mode widget with white text
    appearance_mode_label = ctk.CTkLabel(sidebar_frame, text="Appearance Mode:", anchor="w", text_color="white")
    appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
    
    appearance_mode_optionmenu = ctk.CTkOptionMenu(
        sidebar_frame, values=["Light", "Dark", "System"], command=change_appearance_mode_event, button_color="white", text_color="black"
    )
    appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10, 10))

    # Create UI scaling widget with white text
    scaling_label = ctk.CTkLabel(sidebar_frame, text="UI Scaling:", anchor="w", text_color="white")
    scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
    
    scaling_optionmenu = ctk.CTkOptionMenu(
        sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=change_scaling_event, button_color="white", text_color="black"
    )
    scaling_optionmenu.grid(row=8, column=0, padx=20, pady=(10, 20))
    
    # Create the main content frame with a purple background
    main_frame = ctk.CTkFrame(root, fg_color="#0077B6", border_width=5, corner_radius=10)
    main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    # Add the process buttons with customized gold color and white text
    process_image_button = ctk.CTkButton(
        main_frame, text="Process Single File", command=process_single_file, fg_color="white", text_color="black", font=("Times New Roman", 12, "bold")
    )
    process_image_button.pack(pady=20)

    process_folder_button = ctk.CTkButton(
        main_frame, text="Process Folder", command=process_folder, fg_color="white", text_color="black", font=("Times New Roman", 12, "bold")
    )
    process_folder_button.pack(pady=20)

    show_ocr_button = ctk.CTkButton(
        main_frame, text="Show OCR Extracted Text", command=show_extracted_text, fg_color="white", text_color="black", font=("Times New Roman", 12, "bold")
    )
    show_ocr_button.pack(pady=20)

    root.mainloop()

# Change appearance mode function
def change_appearance_mode_event(mode):
    ctk.set_appearance_mode(mode)

# Change UI scaling function
def change_scaling_event(scaling):
    if scaling == "80%":
        ctk.set_widget_scaling(0.8)
    elif scaling == "90%":
        ctk.set_widget_scaling(0.9)
    elif scaling == "100%":
        ctk.set_widget_scaling(1.0)
    elif scaling == "110%":
        ctk.set_widget_scaling(1.1)
    elif scaling == "120%":
        ctk.set_widget_scaling(1.2)

# Run the GUI
init_gui()

