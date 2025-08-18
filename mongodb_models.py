from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import bcrypt
from datetime import datetime
import re
# Load environment variables
load_dotenv()

# MongoDB connection string from .env
mongo_url = os.getenv("MONGO_DB_URL")
client = MongoClient(mongo_url, server_api=ServerApi('1'))

# Database setup
try:
    client.admin.command('ping')
    print("Pinged your deployment. Successfully connected to MongoDB!")
except Exception as e:
    print("Error connecting to MongoDB:", e)

db = client.my_pdf_query #database name
register = db.user_logins #collection name
register.create_index([("email_id", 1)], unique=True)  # Ensure email_id is unique

def hash_password(plain_password):  
    salt = bcrypt.gensalt() 
    hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), salt) 
    return hashed_password

def check_password(plain_password, hashed_password): 
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)


def insert(first_name: str, last_name: str,email_id:str, password: str):
    if register.find_one({"email_id":email_id}):
        return "Already exists"
    pattern = re.compile(r'^(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$')
    if not pattern.match(password):
        return "Password Format Error"
    try:
        hashedpassword=hash_password(password)
        register.insert_one({"first_name":first_name ,"last_name":last_name,"email_id":email_id, "password": hashedpassword, "created_at": datetime.now()})
        return "Success"
    except Exception as e:
        print("Error inserting data:", e)

def check(email_id: str, password: str):
    user = register.find_one({"email_id": email_id})
    if not user:
        return "No User Exists"
    try:
        if check_password(password,user["password"]):
            return "Success"
        else:
            return "Wrong Password"
    except Exception as e:
        print("Error in password checking function and the error is ",e)
        return "Error"

#test insert function
# if __name__ == "__main__":
#     # Example usage
#     # print(insert("John", "Doe", "john.doe@example.com","Password123!"))
#     print(check("john.doe123@example.com","Password123!"))