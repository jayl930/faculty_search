import pandas as pd
import warnings


# Define the filter function
def filter_langchain_warnings(
    message, category, filename, lineno, file=None, line=None
):
    return "LangChainDeprecationWarning" in str(category)


# Apply the filter for warnings
warnings.showwarning = filter_langchain_warnings
warnings.filterwarnings("ignore")

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# os.environ["AZURE_OPENAI_API_KEY"] =
# os.environ["AZURE_OPENAI_ENDPOINT"] =

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version="2024-03-01-preview",
)
llm = AzureChatOpenAI(
    azure_deployment="gpt4-dpi",
    openai_api_version="2024-03-01-preview",
    temperature=0.1,
)

vector_store = PineconeVectorStore(index_name="dpi", embedding=embeddings)
retriever = vector_store.as_retriever()


def search_researcher_repo(topic, top_k=None):
    if not isinstance(topic, str) or not isinstance(top_k, int):
        raise RuntimeError("Invalid input types")
    if top_k < 0:
        raise RuntimeError("k must be non-negative")

    docs = vector_store.similarity_search_with_score(topic, top_k)
    seen_sources = set()
    results = []

    for doc, score in docs:
        source = doc.metadata["source"]
        source_without_extension = source.split(".pdf")[0]

        if source_without_extension not in seen_sources:
            seen_sources.add(source_without_extension)
            name = " ".join(
                source_without_extension.split("/")[-1].split(".")[0].split("_")
            )
            results.append({"score": score, "Name": name})

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", "Name", "score"]]

    if top_k is not None and len(df) > top_k:
        df = df.head(top_k)

    return df


def search_researcher_data(description, top_k):
    df_openai = search_researcher_repo(description, top_k=top_k)
    df = pd.read_csv("ui.csv")
    df_final = df_openai.merge(df, how="left", on="Name")

    return df_final
