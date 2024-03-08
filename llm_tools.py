import pandas as pd
import pinecone
import boto3
import streamlit as st
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import BedrockEmbeddings
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_json_chat_agent
from langchain.memory import DynamoDBChatMessageHistory
from langchain import hub
from langchain_community.chat_models import BedrockChat

PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
OPENAI_API_KEY = st.secrets.OPENAI_API_KEY

# Initialize clients and services
openai_api_key = OPENAI_API_KEY
session = boto3.Session(region_name='us-east-1')
bedrock_client = boto3.client("bedrock-runtime", region_name='us-east-1')
index_pinecone = 'unidosus-policy-test'
llm_bedrock = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs={"temperature": 0.1})
llm = ChatOpenAI(model_name="gpt-4-0125-preview", openai_api_key=openai_api_key, streaming=True)
embeddings = BedrockEmbeddings(client=bedrock_client, region_name='us-east-1')
history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id="2", boto3_session=session)

def pinecone_db():
    """
    Initializes and returns the Pinecone index.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_pinecone)
    return index

index = pinecone_db()
vectorstore = Pinecone(index, embeddings, "text")
retriever = vectorstore.as_retriever()


def create_filter_conditions(selected_years, options=None):
    """
    Creates filter conditions for document retrieval based on selected years and types.
    """
    filter_conditions = {"year": {"$gte": selected_years[0], "$lte": selected_years[1]}}
    if options and "ALL" not in options:
        filter_conditions["type"] = {"$in": options}
    return filter_conditions

def doc_retrieval(query):
    """
    Answer user's query about UnidosUS
    """
    selected_years = st.session_state.get('selected_years', [2000, 2024])  # Valores predeterminados en caso de que no estén definidos
    selected_types = st.session_state.get('selected_types', ['ALL'])
    filter_conditions = create_filter_conditions(selected_years, selected_types)
    retriever = vectorstore.as_retriever(search_kwargs={'filter': filter_conditions, 'k': 50})
    docs = retriever._get_relevant_documents(query, run_manager=None)
    return docs

uus_tool =Tool(
            name = 'Knowledge Base',
            func = doc_retrieval,
            description = (
                'use this tool to answer questions about UnidosUS or NCLR'
                'more information about the topic'
    )
)
tools = [uus_tool]
prompt = hub.pull("hwchase17/react-chat-json")
chat_agent = create_json_chat_agent(llm=llm_bedrock, tools=tools, prompt = prompt)

