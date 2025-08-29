from fastapi import FastAPI, HTTPException, Request,File,UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mongodb_models import insert, check
from starlette.concurrency import run_in_threadpool
import os
import secrets
import asyncio
import shutil

from doc_retriever import ask_question
from pdf_parsser import store_pdf, parse_pdf, delete_all_traces
from text_embedddings import store_text_and_images,delete_user_embeddings
app = FastAPI()
MAIN_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(MAIN_DIR, "input")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
# Explicitly define allowed origins for CORS
origins = [
    "http://localhost:5173",  # React dev server
    "http://127.0.0.1:5173",  # sometimes needed
    "https://pdf-query-frontend-beta.vercel.app", #my site
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,  # Explicitly list allowed origins
    allow_credentials = True,  # Allow cookies and credentials
    allow_methods = ["*"],  # Allow all HTTP methods
    allow_headers = ["*"],  # Allow all headers
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", secrets.token_hex(16)),
    same_site="none",  # Required for cross-origin cookies
    https_only=True,  # Ensure cookies are sent only over HTTPS
    max_age= 3600,  # Set cookie expiration time (1 hour)
)
@app.middleware("http") 
async def update_session_timeout(request: Request, call_next): 
    response = await call_next(request) 
    if "session" in request.session: 
        # response.set_cookie("session", request.cookies["session"], max_age= 3600, httponly=True, samesite="strict", secure=True) #local
        response.set_cookie(
    key="session",
    value=request.cookies.get("session", ""),  # safer: returns empty string if not present
    max_age=3600,
    httponly=True,
    samesite="none",   # allow same-site requests plus normal form submissions
    secure=True       # required on HTTPS (Azure Web App)
)

 
    return response

class UserDetails(BaseModel):
    first_name: str
    last_name: str
    email_id: str
    password: str

# Endpoints
@app.post("/register/user")
async def create_user(user: UserDetails):
    try:
        value = insert(user.first_name, user.last_name, user.email_id,user.password)
        if value == "Already exists":
            raise HTTPException(status_code=400, detail="User already exists")
        elif value == "Password Format Error":
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters long, contain at least one uppercase letter, one digit, and one special character.")
        return {"message": "Registration successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class Login(BaseModel):
    email_id: str
    password: str

@app.post("/login")
async def login(user: Login, request: Request):
    try:
        result = check(user.email_id, user.password)
        if result == "Success":
            request.session["email_id"] = user.email_id
            await run_in_threadpool(delete_all_traces,user.email_id)
            await run_in_threadpool(delete_user_embeddings,user.email_id)
            return {"message": "Login successful", "email_id": user.email_id}
        elif result == "No User Exists":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        elif result == "Wrong Password":
            raise HTTPException(status_code=401, detail="Invalid credentials")
        elif result == "Error":
            raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        raise HTTPException(status_code=500,detail="Internal server error")

@app.post("/protected/upload/")
async def upload( request: Request,files: list[UploadFile] = File(...)):
    try:
        user_email= request.session.get("email_id")
        if not user_email:
            raise HTTPException(status_code=401, detail="User not logged in")
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        user_directory_info=store_pdf(useremail=user_email)
        user_input_directory=user_directory_info[0]
        all_filenames=[]
        for file in files:
            all_filenames.append(file.filename)
            file_path = os.path.join(user_input_directory, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        await asyncio.to_thread(parse_pdf, dir_list=user_directory_info, useremail=user_email)
        await asyncio.to_thread(store_text_and_images, user_email=user_email)

        # parse_pdf(dir_list=user_directory_info,useremail=user_email)
        # store_text_and_images(user_email=user_email)
        print("Files uploaded and processed successfully.")
        return {"message": "files uploaded and processed.","filenames": all_filenames}
    except Exception as e:
        if request.session.get("email_id"):
            user_email = request.session.get("email_id")
            delete_all_traces(email_address=user_email)
            delete_user_embeddings(user_email=user_email)
            print("Deleted user traces due to error.")
        print("Error in file upload and processing.",e)
        raise HTTPException(status_code=500, detail="An error occurred during file upload and processing.")

class Chat(BaseModel):
    query:str
    image_switch:Optional[bool]=False  

@app.post("/protected/chat/all_pdfs/")
async def chat_with_all_pdfs(chat:Chat,request: Request):
    try:
        user_email = request.session.get("email_id")
        if not user_email:
            raise HTTPException(status_code=401, detail="User not logged in")
        result:dict= await ask_question(user_email=user_email,input=chat.query,
                                 image_search_switch=chat.image_switch)
        response={'message':"Successfully retrieved",
                  'result':result}
        return response
    except Exception as e:
        print("Error in chat with all pdfs is :",e)
        raise HTTPException(status_code=400,detail="Problem with chat")

class ChatSpecificPDFs(BaseModel):
    query: str
    image_switch: Optional[bool] = False
    pdf_names: List[str]

@app.post("/protected/chat/specific_pdfs/")
async def chat_with_specific_pdfs(chat:ChatSpecificPDFs,request: Request):
    try:
        user_email = request.session.get("email_id")
        pdf_names_cleaned=[f.rsplit('.', 1)[0] for f in chat.pdf_names]
        if not user_email:
            raise HTTPException(status_code=401, detail="User not logged in")
        result:dict= await ask_question(user_email=user_email,input=chat.query,
                                 image_search_switch=chat.image_switch,
                                 pdf_to_check_switch=True,
                                 pdf_to_check=pdf_names_cleaned)
        response={'message':"Successfully retrieved",
                  'result':result}
        return response
    except Exception as e:
        print("Error in the chat specific pdf is",e)
        raise HTTPException(status_code=400,detail="Problem with chat")

class ChatOnePDFPage(BaseModel):
    query: str
    image_switch: Optional[bool] 
    pdf_name: str
    page_number: int

@app.post("/protected/chat/one_pdf_page/")
async def chat_with_one_pdf_page(chat:ChatOnePDFPage,request: Request):
    try:
        user_email = request.session.get("email_id")
        # print(chat.pdf_name,chat.page_number)
        pdf_name_store= chat.pdf_name
        pdf_name_cleaned= pdf_name_store.rsplit('.',1)[0]
        if not user_email:
            raise HTTPException(status_code=401, detail="User not logged in")
        result:dict= await ask_question(user_email=user_email,input=chat.query,
                                 image_search_switch=chat.image_switch,
                                 one_pdf_page_check_switch=True,
                                 one_pdf_page_check=[pdf_name_cleaned,chat.page_number])
        response={'message':"Successfully retrieved",
                  'result':result}
        return response
    except Exception as e:
        print("The error in chat with a pdf is :",e)
        raise HTTPException(status_code=400,detail="Problem with chat")   

@app.post("/protected/logout/")
async def logout(request: Request):
    try:
        user_email=request.session.get("email_id")
        if not user_email:
            raise HTTPException(status_code=401, detail="User not logged in")
        await run_in_threadpool(delete_all_traces,user_email)
        await run_in_threadpool(delete_user_embeddings,user_email)
        request.session.pop("email_id", None)
        return {"message": "Logout successful"}
    except Exception as e:
        print("Error in logout is :",e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/health")
async def health():
    return {"message":"Healthy server"}