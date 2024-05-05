from fastapi import FastAPI
from dotenv import load_dotenv
import os
import gradio as gr
import pandas as pd

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()
app = FastAPI()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_EMBEDDING = os.getenv("AZURE_EMBEDDING")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING,
    openai_api_version="2024-03-01-preview",
)
llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    openai_api_version="2024-03-01-preview",
    temperature=0.1,
)
df = pd.read_csv("search_people.csv", header=None)
researcher_names = df[0].str.replace(".pdf", "").tolist()
vector_store = PineconeVectorStore(index_name="dpi", embedding=embeddings)
chain = None

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            As an AI assistant, your primary role is to help me answer questions about UIUC faculties and their researches and experiences.
            The following context is research papers and their abstracts about a single faculty member at UIUC. Make sure to use all context provided. 
            Answer the question using ONLY the following context. 
            If you don't know the answer, just say you don't know. DO NOT make up an answer.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(documents.page_content for documents in docs)


def setup_chain(selected_researcher):
    global chain
    if not selected_researcher:
        return "Please select a researcher."
    filter_param = {"source": f"{selected_researcher}.pdf"}
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10, "filter": filter_param}
    )
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    return "Ready to talk about " + selected_researcher


def on_button_click(selected_researcher):
    setup_message = setup_chain(selected_researcher)
    return setup_message


def chatbot_response(messages, history):
    if chain is None:
        return "No researcher selected. Please select a researcher first."
    response = chain.invoke(messages)
    return response.content


with gr.Blocks() as demo:
    gr.Markdown("# Ask about UIUC researchers")
    gr.Markdown("### Please select a researcher first and ask a question.")
    with gr.Row():
        with gr.Column(scale=1):
            selected_researcher_dropdown = gr.Dropdown(
                label="Select a Researcher", choices=researcher_names
            )
            submit_button = gr.Button("Submit")
            setup_output = gr.Textbox(label="Current Researcher")

        with gr.Column(scale=2):
            chat_interface = gr.ChatInterface(
                fn=chatbot_response,
                # title="DPI Chatbot",
                chatbot=gr.Chatbot(render=False, height=500),
            )

    submit_button.click(
        fn=on_button_click,
        inputs=[selected_researcher_dropdown],
        outputs=[setup_output],
    )

app = gr.mount_gradio_app(app, demo, path="/")
