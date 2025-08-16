import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from typing_extensions import List, TypedDict, Optional
from text_embedddings import search_images, search_text_page_per_pdf, search_text_total_directory,search_text_specific_pdfs, search_images_specific_pdfs 
from image_processing import preprocess_image
import google.generativeai as genai
import io
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a helpful assistant. Answer the question based on the context provided in detail.
                if the context is not sufficient, say 'I don't know'.
                Context:{context}
                Question: {question}
                """)  

class State(TypedDict):
    question: str
    user_email: str
    image_search_switch: Optional[bool]
    pdf_to_check_switch: Optional[bool]
    one_pdf_page_check_switch: Optional[bool]
    pdf_to_check: List[str]
    one_pdf_page_check: list[str,int]
    context: List[Document]
    context_img_path: List[str]
    answer: str
    answer_img: str

def retriever(state: State):
    # 🔹 Get text matches
    if state.get("pdf_to_check_switch", False):
        pdf_names = state.get("pdf_to_check", [])
        if not pdf_names:
            raise ValueError("No PDFs specified for text search.")
        text_results = search_text_specific_pdfs(
            user_email=state["user_email"],
            query=state["question"],
            pdf_names=pdf_names,
            top_k=10
        )
    elif state.get("one_pdf_page_check_switch", False):
        pdf_name, page_number = state.get("one_pdf_page_check", (None, None))
        if not page_number or not pdf_name:
            raise ValueError("Page number and PDF name must be specified for text search.")
        text_results = search_text_page_per_pdf(
            user_email=state["user_email"],
            query=state["question"],
            page_number=page_number,
            pdf_name=pdf_name,
            top_k=10
        )
    else:
        text_results = search_text_total_directory(
            user_email=state["user_email"],
            query=state["question"],
            top_k=10
        )
    # 🔹 Get image matches
    if state.get("image_search_switch", False):
        if state.get("pdf_to_check_switch", False):
            pdf_names = state.get("pdf_to_check", [])
            if not pdf_names:
                raise ValueError("No PDFs specified for image search.")
            img_results = search_images_specific_pdfs(
                user_email=state["user_email"],
                query=state["question"],
                pdf_names=pdf_names,
                top_k=2
            )
        else:
            img_results = search_images(
                user_email=state["user_email"],
                query=state["question"],
                top_k=2
            )
    else:
        img_results = {}


    # Convert text results into serializable dict format
    context = [
        {
            "doc": doc.metadata.get("doc"),
            "page": doc.metadata.get("page"),
            "text": doc.page_content
        }for doc in text_results
    ]

    # Extract image file paths from metadata
    context_img_path = [
        meta.get("image_path")
        for meta in img_results.get("metadatas", [[]])[0]
    ] if img_results and img_results.get("metadatas") else []

    return {
        "context": context,
        "context_img_path": context_img_path
    }

def generate_text_content(state: State):
    docs_content = "\n\n".join(
        f"Document: {doc['doc']}, Page: {doc['page']},\n{doc['text']}"
        for doc in state["context"]
    )
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def generate_img_content(state: State):
    if not state.get("image_search_switch", True):
        return {"answer_img": "Image search is disabled"}
    
    if not state.get("context_img_path"):
        return {"answer_img": "No images exist in this context"}
    
    userquery = f"""Analyze the images given and give a 200 word description of the content in each image.
    The question asked by the user was {state['question']}, use this information as a context.
    If the given image is not related to the context given to you, just output 'No corresponding images to the query'."""


    if not state.get("context_img_path"):
        return {"answer_img": "No images exist in this context"}

    model_img = genai.GenerativeModel("gemini-1.5-flash")

    image_inputs = []
    for path in state["context_img_path"]:
        img = preprocess_image(path)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        image_inputs.append({
            "mime_type": "image/png",
            "data": buf.getvalue()
        })

    input_list = [userquery] + image_inputs
    response = model_img.generate_content(input_list)

    return {"answer_img": response.text}

# Build graph
graph_builder = StateGraph(State).add_sequence([retriever, generate_text_content, generate_img_content])
graph_builder.add_edge(START, "retriever")
graph = graph_builder.compile()

def ask_question(input:str, user_email:str, image_search_switch:bool=False,
                  pdf_to_check_switch:bool=False, one_pdf_page_check_switch:bool=False, 
                  pdf_to_check:list=None, one_pdf_page_check:list=None):
    try:
        print("running ask question")
        if pdf_to_check_switch and  one_pdf_page_check_switch:
            raise ValueError("You can only select one of pdf_to_check_switch or one_pdf_page_check_switch")
        state_input = {
            "question": input,
            "user_email": user_email,
            "image_search_switch": image_search_switch,
            "pdf_to_check_switch": pdf_to_check_switch,
            "one_pdf_page_check_switch": one_pdf_page_check_switch,
            "pdf_to_check": pdf_to_check if pdf_to_check else [],
            "one_pdf_page_check": one_pdf_page_check if one_pdf_page_check else [],
        }
        state_output = graph.invoke(state_input)
        return state_output["answer"], state_output["answer_img"]

    except Exception as e:
        print("Error in bot:", e)
        return False
    
print(ask_question(input="what is RNN", user_email="vijay.anand5306@zoho.com", image_search_switch=True,
                   pdf_to_check_switch=True, pdf_to_check=["Unit 4 NLP"]))
