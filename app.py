# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import pandas as pd
import streamlit as st
import re
from llm_tools import chat_agent,tools,history

# Constants for date range and document options
MIN_YEAR = 2000
MAX_YEAR = 2024
DOCUMENT_TYPES = ['ALL', 'Annual Report','Fact Sheet', 'Article', 'Letter', 'Research Report', 'Appeal Letter', 'Book', 'Other']

def render_search_results(documents):
    """
    Renders search results into a DataFrame for display.
    Acepta una lista de objetos Document, extrayendo metadatos para cada uno.
    """
    metadata_list = []
    for doc in documents:
        # Asumimos que cada 'doc' es un objeto con un atributo 'metadata' accesible
        metadata = doc.metadata  # Accediendo a los metadatos a través del atributo 'metadata'
        
        title = metadata.get('title', '')
        source = metadata.get('source', '').replace('s3://', 'https://s3.amazonaws.com/')
        doc_type = metadata.get('type', '')
        year = metadata.get('year', '')
        if year:
            year = str(int(year))
        metadata_list.append({"Title": title, "Source": source, "Type": doc_type, "Year": year})    
    # Creando y desduplicando DataFrame
    df = pd.DataFrame(metadata_list).drop_duplicates(subset=['Title'])
    return df

def add_response_to_history(response):
    history.add_user_message(response["input"])
    history.add_ai_message(f'"{response["output"]}\"')

def run():
        # Display the application title and caption
    st.set_page_config(page_title="Policy Document Assistant")
    st.title("Policy Document Assistant - v2")

        # Sidebar for filtering documents by time period and type
    with st.sidebar:
        st.image('https://unidosus.org/wp-content/themes/unidos/images/unidosus-logo-color-2x.png', use_column_width=True)
        st.title("Select Time Period")
        selected_years = st.slider("Year", min_value=MIN_YEAR, max_value=MAX_YEAR, value=(2012, 2018), step=1, format="%d")
        st.title("Select Document Type")
        selected_types = st.multiselect('Select Type:', DOCUMENT_TYPES)

    st.session_state['selected_years'] = selected_years
    st.session_state['selected_types'] = selected_types
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

    if prompt := st.chat_input(placeholder="What is UnidosUS?"):
        st.chat_message("user").write(prompt)
        agent_executor = AgentExecutor(
        agent=chat_agent,
        tools=tools, 
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent_executor.invoke({"input": f"{prompt}"},{"callbacks": [st_cb]})
            #st.subheader('Answer:')
            st.write(response["output"])
            #st.write(response["intermediate_steps"][0][1])
            if 'intermediate_steps' in response and len(response["intermediate_steps"]) > 0 and len(response["intermediate_steps"][0]) > 1:
                documents = response["intermediate_steps"][0][1]
                if documents:  # Adicionalmente verifica si la lista de documentos no está vacía
                    df = render_search_results(documents)
                    st.subheader('Sources:')
                    st.data_editor(
                        df,
                        column_config={
                            "Source": st.column_config.LinkColumn("Source")
                        },
                        hide_index=True,
                        ) 
            add_response_to_history(response)           
            #process_and_save_messages(response["chat_history"])                           
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

if __name__ == "__main__":
    run()
