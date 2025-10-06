import fitz  # PyMuPDF
import os
import json
# Set your input and output directories
import re
import shutil

from docx import Document

from AudioConverter import convert_to_wav
from AudioTranscriber import transcribe_full_audio

AUDIO_EXTENSIONS = ('.mp3', '.m4a', '.flac') #wav is natively supported by azure so no need to add here

def safe_folder_name(email: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9]', '_', email.lower())
    return name.strip('_')

def store_pdf(useremail:str)->list:
    main_dir = os.path.dirname(__file__)
    input_pdf_dir = os.path.join(main_dir, "input")
    output_pdf_dir = os.path.join(main_dir, "output")
    if not os.path.exists(input_pdf_dir):
        os.makedirs(input_pdf_dir,exist_ok=True)
    if not os.path.exists(output_pdf_dir):
        os.makedirs(output_pdf_dir,exist_ok=True)
    FolderName:str=safe_folder_name(useremail)
    user_input_dir=os.path.join(input_pdf_dir,FolderName)
    user_output_dir=os.path.join(output_pdf_dir,FolderName)
    os.makedirs(user_input_dir,exist_ok=True)
    os.makedirs(user_output_dir,exist_ok=True)
    return [user_input_dir,user_output_dir]

def parse_pdf(dir_list:list,useremail:str):
    pdf_dir = dir_list[0]  # Folder containing multiple PDFs
    output_dir = dir_list[1]
    os.makedirs(output_dir, exist_ok=True)
    structured_output=[]
    # Loop over all PDF files in the input directory
    for file_name in os.listdir(pdf_dir):
        if file_name.lower().endswith(AUDIO_EXTENSIONS):
            try:
                file_path = os.path.join(pdf_dir, file_name)
                print(f"Processing audio file: {file_path}")
                convert_to_wav(input_file=file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            doc_name = os.path.splitext(pdf_file)[0]
            doc = fitz.open(pdf_path)
            print(f"Processing: {pdf_file}")
            for i, page in enumerate(doc):
                # Extract and append text to the single file

                text = page.get_text()
                page_num = i + 1
                image_filename_list=[]
                # Save images (optional)
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5:  # RGB
                        image_filename = os.path.join(output_dir, f"{doc_name}_page_{i+1}_img_{img_index+1}.png")
                        image_filename_list.append(image_filename)
                        pix.save(image_filename)
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        image_filename = os.path.join(output_dir, f"{doc_name}_page_{i+1}_img_{img_index+1}.png")
                        image_filename_list.append(image_filename)
                        pix.save(image_filename)
                    pix = None  # Free memory
                structured_output.append({
                    "user_email":useremail,
                    "doc": doc_name,
                    "page": page_num,
                    "text": text.strip(),
                    "image_file_name":image_filename_list})
            doc.close()
        
        elif pdf_file.endswith(".docx"):
            docx_path = os.path.join(pdf_dir, pdf_file)
            doc_name = os.path.splitext(pdf_file)[0]
            docx_file = pdf_file

            print(f"Processing: {docx_file}")

            # Load the document
            try:
                doc = Document(docx_path)
            except Exception as e:
                print(f"❌ Error reading {docx_file}: {e}")
                continue

            # Extract text
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)

            # Extract images
            image_filename_list = []
            image_count = 0
            for rel in doc.part.rels:
                rel = doc.part.rels[rel]
                if "image" in rel.target_ref:
                    image_count += 1
                    image_data = rel.target_part.blob
                    image_filename = f"{doc_name}_img_{image_count}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    image_filename_list.append(image_path)

            # Create one record per document (like one “page” for PDF version)
            structured_output.append({
                "user_email": useremail,
                "doc": doc_name,
                "page": 1,  # No concept of pages in DOCX
                "text": text,
                "image_file_name": image_filename_list
            })
        elif pdf_file.lower().endswith('.wav'):
            try:
                audio_path =os.path.join(pdf_dir, pdf_file)
                audio_name = os.path.splitext(pdf_file)[0]
                print(f"Processing audio file: {audio_path}")
                audio_text = transcribe_full_audio(audio_path)
                structured_output.append({
                    "user_email": useremail,
                    "doc": audio_name,
                    "page": 1,
                    "text": audio_text,
                    "image_file_name": []
                })
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
    # Save the structured output to a JSON file

    json_path = os.path.join(output_dir, "all_text.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, ensure_ascii=False, indent=2)
    print("Done JSON dumping")
    print(f"Done! Combined text and images saved in: {os.path.abspath(output_dir)}")

def delete_all_traces(email_address:str=None):
    if email_address is None:
        raise ValueError("Email address must be provided to delete traces.")
    user_folder = safe_folder_name(email_address)
    main_dir = os.path.dirname(__file__)
    input_dir = os.path.join(main_dir, "input")
    output_dir = os.path.join(main_dir, "output")
    input_directory = os.path.join(input_dir, user_folder)
    output_directory = os.path.join(output_dir, user_folder)
    if os.path.exists(input_directory):
        shutil.rmtree(input_directory)
        print(f"Deleted folder: {input_directory}")

    # Delete output folder
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        print(f"Deleted folder: {output_directory}")

# delete_all_traces("vijay.anand5306@zoho.com")
# store=store_pdf("vijay.anand5306@zoho.com")
# parse_pdf([r"input\vijay_anand5306_zoho_com",r"output\vijay_anand5306_zoho_com"],"vijay.anand5306@zoho.com")
#Testing Done

