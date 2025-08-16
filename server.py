from fastapi import FastAPI, HTTPException, Request,Query,Depends,File,UploadFile
from pydantic import BaseModel
from typing import Optional, List
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pdf_parsser import parse_pdf,delete_all_traces,store_pdf
import os
import shutil
from doc_retriever import ask_question
app = FastAPI()
UPLOAD_DIR=r"c:\Users\vijay\Documents\Programming\docu_retiever_langchain\input"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(input, exist_ok=True)
# Explicitly define allowed origins for CORS
allowed_origins = ["*"]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Explicitly list allowed origins
    allow_credentials=True,  # Allow cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
class Chat(BaseModel):
    query:str

@app.post("/chat/")
async def chatables(chat:Chat):
    try:
        result:tuple=ask_question(chat.query)
        response={'message':"Successfully retrieved",
                  "text_answer":result[0],
                  "img_answer":result[1]}
        return response
    except:
        raise HTTPException(status_code=400,detail="Problem with chat")

@app.post("/upload/")
async def upload(user_email:str,files: list[UploadFile] = File(...)):
    try:
        # delete_all_traces()
        user_directory_info=store_pdf(useremail=user_email)
        user_input_directory=user_directory_info[0]
        user_output_directory=user_directory_info[1]
        for file in files:
            file_path = os.path.join(user_input_directory, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        parse_pdf(dir_list=user_directory_info,useremail=user_email)
        return {"message": "files uploaded and processed."}
    except:
        delete_all_traces()
        raise HTTPException(status_code=400,detail="problem with the files")
@app.get("/health")
async def health():
    return {"message":"Healthy server"}