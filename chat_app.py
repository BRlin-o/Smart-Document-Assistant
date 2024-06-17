from pprint import pprint
import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationChain
from typing import List, Tuple, Union, Dict
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor, create_react_agent

from utils.streamlit import StreamHandler, display_chat_messages, langchain_messages_format, render_chat_interface
from utils.rag import get_rag_chain
from utils.prompt import GOGORO_PROMPT

from dotenv import load_dotenv
load_dotenv()

llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct", 
    # model="nvidia/nemotron-4-340b-instruct", 
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"), 
    temperature=0.0,
    top_p=1,
    max_tokens=1024,
    streaming=True,
    stream=True,
    verbose=True,
    # callbacks=[StreamHandler(st.empty())]
)

memory = ConversationBufferWindowMemory(
    k=10,
    ai_prefix="Assistant",
    human_prefix="Hu",
    chat_memory=StreamlitChatMessageHistory(),
    return_messages=False,
    memory_key="chat_history",
    input_key="input"
)

def call_gogoro_search(query):
    print("[DEBUG] call_gogoro_search query", query)
    return get_rag_chain(llm, st.session_state["car_model"].upper(), retriever_k=5)(f"{query}?")

LLM_AGENT_TOOLS = [
    Tool(
        name="GogoroSearch",
        func=lambda query: call_gogoro_search(query),
        description=(
            "Use when you are asked any questions about Gogoro."
            " The Input should be a correctly formatted question, and using 繁體中文."
            " If the response contains any markdown syntax like `![]()`, please make sure to display it directly in the chat, as the frontend can render markdown to show image data."
        ),
    )
]

# agent = create_tool_calling_agent(
agent = create_react_agent(
    llm=llm, 
    tools=LLM_AGENT_TOOLS, 
    prompt=GOGORO_PROMPT
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=LLM_AGENT_TOOLS,
    verbose=True,
    return_intermediate_steps=False,
    memory=memory,
    # handle_parsing_errors="Check your output and make sure it conforms!",
    # handle_parsing_errors=True,
)

from langchain_core.runnables import RunnableLambda

get_output = RunnableLambda(lambda x: x["output"])

chain = agent_chain | get_output

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # AIMessage(content="Hello, I am your assistant. How can I help you? ![output_test](app/static/CrossOver/9_image_1.png)"),
        AIMessage(content="Hello, I am your assistant. How can I help you?"),
    ]

if __name__ == "__main__":
    car_model, chat_lang = render_chat_interface()
    display_chat_messages()

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

    st.session_state["langchain_messages"] = langchain_messages_format(
        st.session_state["langchain_messages"]
    )

    if isinstance(st.session_state.chat_history[-1], AIMessage) is False:
        with st.chat_message("AI"):
            response = chain.invoke(
                {
                    "chat_history": st.session_state.chat_history,
                    "input": user_query,
                    "car_model": car_model, 
                    "language": chat_lang
                    # "language": st.session_state["chat_lang"]
                },
                {
                    "callbacks": [
                        # StreamHandler(st.empty()),
                        StreamlitCallbackHandler(st.container())
                    ]
                },
            )
            print("[DEBUG] response:")
            pprint(response)
            message = AIMessage(content=response)
            st.session_state.chat_history.append(message)
            st.rerun() 
        
## 你能幫我甚麼?
## 智慧鑰匙卡感應器的位置在哪裡？