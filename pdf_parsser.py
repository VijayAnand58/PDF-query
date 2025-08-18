import fitz  # PyMuPDF
import os
import json
# Set your input and output directories
import re
import shutil

def safe_folder_name(email: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9]', '_', email.lower())
    return name.strip('_')

def store_pdf(useremail:str)->list:
    main_dir = os.path.dirname(__file__)
    input_pdf_dir = os.path.join(main_dir, "input")
    output_pdf_dir = os.path.join(main_dir, "output")
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
    # Loop over all PDF files in the input directory
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            doc_name = os.path.splitext(pdf_file)[0]
            doc = fitz.open(pdf_path)
            structured_output=[]

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

