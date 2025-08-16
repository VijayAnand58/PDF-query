import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

PERSIST_DIR = r"C:\Users\vijay\Documents\Programming\docu_retiever_langchain\chromadb"

# LangChain text embeddings
text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# CLIP model for images
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def store_text_and_images(parsed_json_path):
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Prepare documents for text collection
    docs = []
    image_data = []

    for entry in entries:
        meta = {
            "user_email": entry["user_email"],
            "doc": entry["doc"],
            "page": entry["page"]
        }

        if entry["text"].strip():
            docs.append(Document(page_content=entry["text"], metadata=meta))

        for img_path in entry["image_file_name"]:
            if os.path.exists(img_path):
                image_data.append((img_path, meta))

    # --- Chunk and store text ---
    separators = ["\n### ", "\n\n", "\n", ". ", " ","" ]
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, separators=separators ,chunk_overlap=100)
        chunked_docs = splitter.split_documents(docs)

        Chroma.from_documents(
            documents=chunked_docs,
            embedding=text_embeddings,
            persist_directory=PERSIST_DIR,
            collection_name="tenant_embeddings_text"
        ).persist()

    # --- Store image embeddings ---
    if image_data:
        img_store = Chroma(
            collection_name="tenant_embeddings_img",
            persist_directory=PERSIST_DIR
        )

        img_embeddings = []
        img_docs = []
        img_metas = []

        for img_path, meta in image_data:
            image = Image.open(img_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)[0].cpu().numpy().tolist()

            img_embeddings.append(emb)
            img_docs.append(os.path.basename(img_path))
            img_metas.append({**meta, "image_path": img_path})

        img_store._collection.add(
            embeddings=img_embeddings,
            documents=img_docs,
            metadatas=img_metas,
            ids=[f"{m['user_email']}_{m['doc']}_p{m['page']}_{d}" for m, d in zip(img_metas, img_docs)]
        )
        img_store.persist()

    print("✅ Data stored in Chroma")

def search_text_total_directory(user_email, query, top_k=5):
    store = Chroma(
        collection_name="tenant_embeddings_text",
        persist_directory=PERSIST_DIR,
        embedding_function=text_embeddings
    )
    return store.similarity_search(query, k=top_k, filter={"user_email": user_email})

def search_text_specific_pdfs(user_email, query, pdf_names, top_k=5):
    store = Chroma(
        collection_name="tenant_embeddings_text",
        persist_directory=PERSIST_DIR,
        embedding_function=text_embeddings
    )
    return store.similarity_search(query, k=top_k, filter={
            "$and": [
                {"user_email": user_email},
                {"doc": {"$in": pdf_names}}
            ]})
    
def search_text_page_per_pdf(user_email, query, page_number, pdf_name ,top_k=5):
    store = Chroma(
        collection_name="tenant_embeddings_text",
        persist_directory=PERSIST_DIR,
        embedding_function=text_embeddings)
    return store.similarity_search(query, k=top_k, filter={
            "$and": [
                {"user_email": user_email},
                {"page": page_number},
                {"doc": pdf_name}
            ]})

def search_images(user_email, query, top_k=5):
    inputs = clip_processor.tokenizer([query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_emb = clip_model.get_text_features(**inputs)[0].cpu().numpy().tolist()

    store = Chroma(
        collection_name="tenant_embeddings_img",
        persist_directory=PERSIST_DIR)
    return store._collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={"user_email": user_email})

def search_images_specific_pdfs(user_email, query, pdf_names, top_k=5):
    inputs = clip_processor.tokenizer([query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_emb = clip_model.get_text_features(**inputs)[0].cpu().numpy().tolist()

    store = Chroma(
        collection_name="tenant_embeddings_img",
        persist_directory=PERSIST_DIR)
    return store._collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={
            "$and": [
                {"user_email": user_email},
                {"doc": {"$in": pdf_names}}
            ]
        })
def delete_user_embeddings(user_email):
    store = Chroma(
        collection_name="tenant_embeddings_text",
        persist_directory=PERSIST_DIR,
        embedding_function=text_embeddings
    )
    store._collection.delete(where={"user_email": user_email})

    img_store = Chroma(
        collection_name="tenant_embeddings_img",
        persist_directory=PERSIST_DIR
    )
    img_store._collection.delete(where={"user_email": user_email})

    print(f"✅ Deleted embeddings for user: {user_email}")


#testing the functions
# if __name__ == "__main__":
# #     # Example usage
#     delete_user_embeddings("vijay.anand5306@zoho.com")
#     parsed_json_path = r"output\vijay_anand5306_zoho_com\all_text.json"
#     store_text_and_images(parsed_json_path)

#     user_email = "vijay.anand5306@zoho.com"
#     query = "What about RNN?"
#     query_img="What about RNN?"
#     text_results = search_text_total_directory(user_email=user_email, query=query, top_k=7)
#     img_results = search_images(user_email, query_img)
#     print("Text Results:")
#     print(text_results)
#     print(img_results)