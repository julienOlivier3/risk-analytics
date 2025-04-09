import operator
from typing import Annotated, List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

# Vector Storage
faiss_vector_store = FAISS.load_local(
    "./faiss_vector_store",
    embeddings=HuggingFaceEmbeddings(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    ),
    allow_dangerous_deserialization=True
)

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Web search tool
tavily_search = TavilySearchResults(max_results=2)

dax_40 = [
    'adidas', 'airbus', 'allianz', 'basf', 'bayer', 'beiersdorf', 'bmw',
    'brenntag', 'commerzbank', 'continental', 'daimler-truck',
    'deutsche-bank', 'deutsche-boerse', 'deutsche-post',
    'deutsche-telekom', 'eon', 'fresenius', 'fresenius medical care',
    'hannover rück', 'heidelberg materials', 'henkel',
    'infineon technologies', 'mercedes benz', 'merck',
    'mtu', 'münchener rück', 'porsche', 'porsche automobil holding',
    'qiagen', 'rheinmetall', 'rwe', 'sap', 'sartorius', 'siemens',
    'siemens energy', 'siemens healthineers', 'symrise',
    'volkswagen', 'vonovia', 'zalando']


class Company(BaseModel):
    name: str = Field(description="Name of DAX 40 company.")
    year: int = Field(2023, description="Year of the annual report.")


class Companies(BaseModel):
    companies: Optional[List[Company]] = Field(
        description="List of DAX 40 companies with information about their annual reports.")


class Topic(BaseModel):
    topic: str = Field(description="Topic of the question.")


class OverallState(MessagesState):
    retrievals: Annotated[List[str], operator.add]
    context: Annotated[List[str], operator.add]  # Accumulated context from all companies
    companies: Companies
    final_answer: str


class DocumentState(MessagesState):
    company: Company  # The company related to the document (single value)
    topic: str  # The topic extracted for the company (single value)


supervisor_instruction = """You are tasked with analyzing whether the following user prompt relates to one or several
 DAX 40 companies: {message}.

If you find any DAX 40 companies mentioned in the user prompt, map them to the ones mentioned in the following list:
{dax_40}.

It may well be that the user prompt is a more generic question that is not related to any specific company.
"""

topic_extraction_instruction = """You are part of an AI agent designed to answer questions about the risks DAX 40
 companies are facing.

Your task is to extract the topic of the user prompt. The topic should be a short phrase that summarizes the main
 subject of the question.

Please make sure to retain specific keywords that are relevant to the topic.

This is the user prompt from which you should extract the topic: {message}.

Company names should not be included in the topic.
"""

single_answer_generation_instruction = """Based on the following context for {company}, generate an answer to the
 topic '{topic}': \n\n {context}.

If the context does not provide enough information, please answer that the annual report of {company} does not provide
 any information about the topic '{topic}'.
"""

final_answer_generation_instruction = """You are tasked to provide a concise answer to the following prompt: {message}.

To respond to the user, you are supposed to use the following contextual information: {context}.

If you find that the context contains repetitive information, please summarize it accordingly.

If the context mentions DAX 40 companies, please make sure to explicitly include them in your answer.
"""


def extract_companies(state: OverallState, dax_40: list = dax_40):

    messages = state.get('messages')
    dax_40 = ', '.join(dax_40)

    # Enforce structured output
    structured_llm = llm.with_structured_output(Companies)

    # System message
    system_message = supervisor_instruction.format(message=messages[-1].content, dax_40=dax_40)

    # Extract companies
    companies = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(
        content="Extract the set of DAX 40 companies if there any mentioned. Otherwise return nothing.")])

    return {'companies': companies.companies}


def route_to_tool(state: OverallState):

    companies = state.get('companies', None)

    if companies is not None:
        messages = state.get('messages')

        structured_llm = llm.with_structured_output(Topic)
        system_message = topic_extraction_instruction.format(message=messages[-1].content)
        topic = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(
            content="Extract the topic of the user prompt.")])

        return [Send("rag_agent", {"company": c, "topic": topic}) for c in companies]

    else:
        return 'web_agent'


def web_agent(state: OverallState):

    messages = state.get('messages')

    search_docs = tavily_search.invoke(messages[-1].content)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def rag_agent(state: DocumentState, vector_store=faiss_vector_store):

    company = state.get('company')
    topic = state.get('topic')

    context = vector_store.similarity_search(
        query=topic.topic,
        k=2,
        filter={"company": company.name, "year": 2023})

    if len(context) == 0:
        answer = f"""
            The annual report of {company.name} does not provide any information about the
            topic '{topic.topic}'."""

    else:
        system_message = single_answer_generation_instruction.format(
            company=company.name,
            topic=topic.topic,
            context=' \n '.join([node.page_content for node in context]))
        answer = llm.invoke(system_message).content

    return {"retrievals": [context], "context": [answer]}


def create_answer(state: OverallState):

    messages = state.get('messages')
    context = state.get('context', None)

    system_message = final_answer_generation_instruction.format(
        message=messages[-1].content,
        context='\n\n'.join(context))
    final_answer = llm.invoke(system_message).content

    # Return the company-specific answer
    return {"final_answer": final_answer}


builder = StateGraph(OverallState)
builder.add_node("extract_companies", extract_companies)
builder.add_node("rag_agent", rag_agent)
builder.add_node("web_agent", web_agent)
builder.add_node("create_answer", create_answer)
builder.add_edge(START, "extract_companies")
builder.add_conditional_edges("extract_companies", route_to_tool, ["rag_agent", "web_agent"])
builder.add_edge("web_agent", "create_answer")
builder.add_edge("rag_agent", "create_answer")
builder.add_edge("create_answer", END)

graph = builder.compile()
