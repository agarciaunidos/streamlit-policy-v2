
import streamlit as st
import pandas as pd
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.memory import DynamoDBChatMessageHistory
import boto3

PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
BEDROCK_REGION = st.secrets.AWS_DEFAULT_REGION
MAX_TOKENS = 10024
TEMPERATURE = 0.7

# Initialize clients and services
session = boto3.Session(region_name='us-east-1')
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
chat_history_DB = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="1", boto3_session=session)
index_pinecone = 'unidosus-policy-test'
model_id = "anthropic.claude-v2:1"
model_kwargs = {"max_tokens_to_sample": MAX_TOKENS, "temperature": TEMPERATURE}
embeddings = BedrockEmbeddings(client=bedrock_client, region_name=BEDROCK_REGION)
llm = Bedrock(model_id=model_id, region_name=BEDROCK_REGION, client=bedrock_client, model_kwargs=model_kwargs)

def pinecone_db():
    """
    Initializes and returns the Pinecone index.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_pinecone)
    return index

def retrieval_answer(query, selected_years, types):
    """
    Retrieves answers and sources based on the query, selected years, and document types.
    """
    # Construct filter conditions for the query
    filter_conditions = create_filter_conditions(selected_years, types)
    index = pinecone_db()
    vectorstore = Pinecone(index, embeddings, "text")
    retriever = vectorstore.as_retriever(search_kwargs={'filter': filter_conditions, 'k': 20})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # Include filter conditions in the prompt for enhanced context
    # Enhance the query with filter details
    filter_details = f"Applying filters: Years between {selected_years[0]} and {selected_years[1]}, Types: {', '.join(types) if types else 'All types'}"
    response = retrieval_chain.invoke({"input": f"{query}", "filter": f"{filter_details}"})
    sources = render_search_results(response['context'])
    # Update chat history in DynamoDB
    chat_history_DB.add_user_message(query)
    ai_message = extract_answer_sources(response)
    chat_history_DB.add_ai_message(ai_message)
    return response['answer'], sources

def create_filter_conditions(selected_years, options=None):
    """
    Creates filter conditions for document retrieval based on selected years and types.
    """
    filter_conditions = {"year": {"$gte": selected_years[0], "$lte": selected_years[1]}}
    if options and "ALL" not in options:
        filter_conditions["type"] = {"$in": options}
    return filter_conditions


def render_search_results(documents):
    """
    Renders search results into a DataFrame for display.
    """
    metadata_list = []
    for doc in documents:
        # Obtenemos los metadatos básicos
        title = doc.metadata.get('title', '')
        source = doc.metadata.get('source', '').replace('s3://', 'https://s3.amazonaws.com/')
        doc_type = doc.metadata.get('type', '')
        year = doc.metadata.get('year', '')

        if year:
            year = str(int(year))

        metadata_list.append({"Title": title, "Source": source, "Type": doc_type, "Year": year})
    df = pd.DataFrame(metadata_list).drop_duplicates(subset=['Title'])
    return df

def extract_answer_sources(data):
    """
    Extracts and formats the answer and its sources from the response data.
    """
    answer = data.get('answer', '')
    sources = [document.metadata['source'] for document in data.get('context', []) if hasattr(document, 'metadata')]
    sources_str = "','".join(sources)
    result = f"{answer}source:'{sources_str}'."
    return result

# Define the prompt template for user queries
PROMPT_TEMPLATE = """
    Answer the following questions as best you can but speaking as assistant expert in summarization and gathering ideas.
    You have access to a vector database where are storage UnidosUS documents, and the user applied the following filters:

    {filter}

    Use the following format:

    Query:
        the input query you must answer.
    Thought:
        you should always think about what to do, based on what the user is asking.
    Action:
        the action to take, should be to use this context from Vector Database, Remember the user may applied filters on it:
<context>{context}</context>.
    Action Input:
        Vector database context.
    Observation:
        the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat up to 3 times)
    Thought:
        I now know the final answer.
    Final Answer:
        The answer should be comprehensive and add the more possible detail Based on the limited excerpt provided you may get.

    Do not make up any answer!
    JUST RESPONSE THE ANSWER!, DO NOT INCLUDE query,thought,action, etc.....
    User's Request: {input} 
    """

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
document_chain = create_stuff_documents_chain(llm, prompt)

