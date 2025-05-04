import os
import re
import subprocess
from typing import Any, Dict, List, Literal, Optional, Union

import faiss
import pandas as pd
from IPython.display import HTML, display
from langchain.schema import Document as LangchainDocument
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (HuggingFaceEmbeddings,
                                   HuggingFaceEndpointEmbeddings)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from llama_index.core import Document
from llama_index.core import Document as LlamaDocument
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (SemanticSplitterNodeParser,
                                          SentenceSplitter)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def print_graph_propagation(graph: CompiledStateGraph, query: str) -> None:
    """
    Prints the propagation of events in a graph based on a user query.

    Parameters:
    graph (object): An object representing the graph from which events are streamed.
    query (str): The user query to filter the graph events.

    Returns:
    None: This function does not return a value; it prints output directly.
    """

    print(f"User query: {query}")
    print("\n-----------------\n")
    print("Graph events:")
    for index, event in enumerate(graph.stream({"messages": query})):
        node = list(event.keys())[0]
        state = list(list(event.values())[0].keys())[0]
        content = str(list(list(event.values())[0].values())[0])
        print(f"{index+1}. Node: {node}")
        print(f"    State: {state}")
        print(f"        Content: {content[:200]} ...")
    print("\n-----------------\n")
    try:
        print(f"Answer: {event['generate_answer']['final_answer'][:200]} ...")
    except KeyError:
        print(f"Answer: {event['off_topic_response']['final_answer'][:200]} ...")


def ollama_list() -> List[str]:
    """
    Run the 'ollama list' command to retrieve a list of available models.

    Returns:
        List[str]: A list of model names as strings.
    """
    # Run the ollama list command and capture the output
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    models = result.stdout.splitlines()  # Split the output into lines
    return models


def ollama_pull(model_name: str) -> None:
    """
    Pull a specified model using the 'ollama pull' command.

    Args:
        model_name (str): The name of the model to pull.

    Returns:
        None: This function does not return a value. It prints the result of the operation.

    Raises:
        subprocess.CalledProcessError: If the command fails to execute.
    """
    try:
        # Run the ollama pull command
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"Model '{model_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull model '{model_name}': {e}")


def get_llm_model(
        llm_type: Literal["openai", "ollama"] = "ollama",
        model_name: str = "llama3.1:8b") -> Optional[ChatOpenAI | ChatOllama]:
    """
    Retrieve and load a language model based on the specified type.

    Args:
        llm_type (Literal["openai", "ollama"]): The type of language model to load.
            Defaults to "ollama".
        model_name (str): The name of the model to load. Defaults to "llama3-1:8b".

    Returns:
        Optional[ChatOpenAI | ChatOllama]: The loaded language model instance (either
        ChatOpenAI or ChatOllama) if successful, or None if the model could not be loaded.

    Raises:
        ValueError: If the specified model name is not found in the available models.
    """
    if llm_type == "openai":
        model = ChatOpenAI(model=model_name, temperature=0.0, api_key=os.getenv("OPENAI_API_KEY", None))
        print(f"{model.model_name} model loaded.")
        return model
    else:
        available_models = ollama_list()
        available_models = [re.split(r'\s{2,}', x) for x in available_models]
        df_available_models = pd.DataFrame(available_models[1:], columns=available_models[0])

        if model_name in df_available_models['NAME'].values:
            model = ChatOllama(model=model_name, temperature=0.0)
            print(f"{model.model} model loaded.")
            return model
        else:
            print(f"Model '{model_name}' not found. Pulling the model which may take a while...")
            ollama_pull(model_name=model_name)
            try:
                model = ChatOllama(model=model_name, temperature=0.0)
                print(f"'{model.name}' model loaded.")
                return model
            except Exception as e:
                print(f"Failed to load model '{model_name}': {e}")
                return None


async def vectorize_chunks(
    chunks: List[Document],
    model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
    device: str = 'cpu',
    inference_api: bool = True,
    huggingfacehub_api_token: Optional[str] = None
) -> FAISS:
    """
    Vectorizes a list of document chunks and stores them in a FAISS vector store.

    This function takes a list of document chunks, converts them into vector embeddings
    using a specified Hugging Face embedding model, and stores the embeddings in a FAISS
    vector store. It supports both local embedding models and Hugging Face Inference API.

    Args:
        chunks (List[Document]): A list of document chunks to be vectorized.
        model_name (str): The Hugging Face embedding model to use. Defaults to
                           "Snowflake/snowflake-arctic-embed-l-v2.0".
        device (str): The device to run the embedding model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        inference_api (bool): Whether to use the Hugging Face Inference API for embeddings.
                              Defaults to True.
        huggingfacehub_api_token (Optional[str]): The Hugging Face Hub API token for authentication.

    Returns:
        FAISS: A FAISS vector store containing the vectorized document chunks.
    """

    embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )

    embed_model_inference = HuggingFaceEndpointEmbeddings(
            model=model_name,
            model_kwargs={'device': device},
            huggingfacehub_api_token=huggingfacehub_api_token
        )

    if inference_api:
        try:
            index = faiss.IndexFlatL2(len(embed_model_inference.embed_query("hello world")))
            embed_model = embed_model_inference
        except Exception as e:
            print(f"Error initializing FAISS index: {e}. Switch to local model.")
            index = faiss.IndexFlatL2(len(embed_model.embed_query("hello world")))
    else:
        index = faiss.IndexFlatL2(len(embed_model.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embed_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    await vector_store.aadd_documents(documents=chunks)

    return vector_store


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = ' ',
    paragraph_separator: str = '\n\n\n',
    semantic_chunking: bool = False,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
    embed_model: Optional[HuggingFaceEmbedding] = None):

    if not semantic_chunking:
        splitter = SentenceSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
            include_metadata=True)
    else:
        if embed_model is None:
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model)

    chunks = splitter.get_nodes_from_documents(document)

    return chunks


def parse_document(
        report_file_path: str,
        pre_process_dict: Optional[Dict[str, Dict[str, Union[range, str]]]] = None,
        company: str = None,
        year: int = None) -> List[Document]:
    """
    Pre-processes a document by loading it from the specified file path, optionally removing specified strings,
    and returning a list of Document objects containing the processed text.

    Args:
        report_file_path (str): The path to the report file to be processed.
        pre_process_dict (Optional[Dict[str, Dict[str, Union[range, str]]]]):
            A dictionary containing optional parameters for the pre-processing of different entities.
            Each key corresponds to an entity (e.g., 'adidas', 'allianz') and maps to another dictionary with:
            - 'pages' (range): A range of page indices to process.
            - 'string_to_remove' (str): A regex pattern for strings to be removed from the document text.

    Returns:
        List[Document]: A list containing a single Document object with the processed text.
    """

    documents = SimpleDirectoryReader(input_files=[report_file_path]).load_data(show_progress=False)

    overall_text = ""

    if pre_process_dict is not None:
        string_to_remove = pre_process_dict.get('string_to_remove')
        pages = pre_process_dict.get('pages', range(0, len(documents), 1))  # Default to all pages if not specified

        for page in pages:
            document = documents[page]
            text = document.text
            if string_to_remove is not None:
                text = re.sub(string_to_remove, "", text)
            overall_text = "\n".join([overall_text, text])
    else:
        # If no pre_process_dict is provided, process all pages
        for page in range(len(documents)):
            document = documents[page]
            overall_text = "\n".join([overall_text, document.text])

    documents = [Document(text=overall_text, metadata={'company': company, 'year': year})]

    return documents


pre_process_dict = {
    "adidas": {
        "pages": range(165, 187, 1),
        "string_to_remove": """1 2 3 4 5 \nT O  O U R SHA REHO L D ERS  GRO U P  MAN A GEMEN
         T  REP O RT – \nO U R CO MPA N Y \nGRO U P  MAN A GEMEN T  REP O RT – \nF I N A N CI
          AL  REVI EW  \nCO N SO L I DA T ED  FI N AN CI A L \nST A T EMEN T S \nA D D I T I
           ON A L I N FO RMA T I ON \n \n\\d{1,3} \n    \n    \n A N N U A L  R E P O R T  2 0 2 3""",
    },
    "allianz": {
        "pages": range(21, 37, 1),
        "string_to_remove": "B _ Management Report of Allianz SE \n\\d{1,3} Annual Report 2023 – Allianz SE \n"
    },
    "basf": {
        "pages": range(172, 183, 1),
        "string_to_remove": "Combined Management’s Report – Opportunities and Risks\n"
    },
    "bayer": {
        "pages": range(99, 116, 1),
        "string_to_remove": """ \n \nBayer Annual Report 2023 A Combined Management Report
        \n3.2 Opportunity and Risk Report\n \\d{1,3}\n"""
    },
    "beiersdorf": {
        "pages": range(155, 166, 1),
        "string_to_remove": """ \n \nBeiersdorf Annual Report 2023 A Combined Management
         Report\n3.2 Opportunity and Risk Report\n \\d{1,3}\n"""
    },
    "bmw": {
        "pages": range(126, 142, 1),
        "string_to_remove": """\\d{1,3} BMW Group Report 2023\\s+To Our Stakeholders
        \\s+Combined Management Report\\s+Group Financial Statements\\s+Responsibility
         Statement and Auditor’s Report\\s+Remuneration Report\\s+Other Information
        \\s+\n\\s+Risks and Opportunities\\s+\n"""
    }
}


def convert_llama_to_langchain(llama_doc: LlamaDocument) -> LangchainDocument:
    """
    Convert a LlamaDocument to a LangchainDocument.

    This function takes a LlamaDocument instance and converts it into a
    LangchainDocument instance, preserving the document's ID, text content,
    and metadata.

    Parameters:
    llama_doc (LlamaDocument): The LlamaDocument instance to be converted.

    Returns:
    LangchainDocument: A LangchainDocument instance containing the same
                       content and metadata as the input LlamaDocument.
    """
    return LangchainDocument(
        id=llama_doc.id_,
        page_content=llama_doc.text,
        metadata=llama_doc.metadata
    )


def display_document_with_image_side_by_side(document: Document, image_path: str) -> None:
    """
    Display the text of a document and an image side by side.

    Args:
        doc_index (Document): llama_index Document.
        image_path (str): Path to the image file to be displayed alongside the document text.
    """
    # Get the text of the document
    document_text = document.text

    # Create HTML content
    html_content = f"""
    <div style="display: flex; align-items: flex-start;">
        <div style="flex: 1; padding: 10px;">
            <pre>{document_text}</pre>
        </div>
        <div style="flex: 1; padding: 10px;">
            <img src="{image_path}" style="max-width: 100%; height: auto;">
        </div>
    </div>
    """

    # Display the HTML content
    display(HTML(html_content))


def print_best_models(results: Dict[str, Dict[str, Any]], model: str) -> None:
    """
    Prints the best model's score and parameters from the results of a hyperparameter search.

    Parameters:
    results (Dict[str, Dict[str, Any]]): A dictionary containing the results of the model search,
                                          where each key is a model name and the value is another
                                          dictionary with model evaluation metrics.
    model (str): The name of the model for which to print the best score and parameters.

    Returns:
    None: This function does not return a value; it prints the results directly.
    """
    model_results = results[model]
    id = model_results['rank_test_score'].argmin()

    score = model_results['mean_test_score'][id]*-1
    params = model_results['params'][id]

    print(f"""Model: {model} \nBest score: {score:.2f} RMSE \nBest parameters: {params} \n""")
