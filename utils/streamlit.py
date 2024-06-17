import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Tuple, Union, Dict

class StreamHandler(BaseCallbackHandler):
    """
    Callback handler to stream the generated text to Streamlit.
    Callback handler 用於將生成的文本流式傳輸到 Streamlit。
    """

    def __init__(self, container: st.container, initial_text: str="") -> None:
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Append the new token to the text and update the Streamlit container.
        將新令牌附加到文本並更新 Streamlit 容器。
        """
        self.text += token
        self.container.markdown(self.text, unsafe_allow_html=True)

def render_chat_interface():
    """
    渲染聊天界面。
    """
    st.title("Smart Document Assistant: A Case Study with Gogoro Vehicle Manuals🏍️") # Gogoro Smart Scooter 萬事通
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            car_model = st.selectbox(
                "車型(Vehicle)", 
                ["CrossOver", "Delight", "JEGO", "SuperSport", "VIVA MIX", "VIVA XL", "VIVA", "S1", "S2", "S3"],
            )
            car_model = car_model.upper()
            st.session_state["car_model"] = car_model
        with col2:
            chat_lang = st.selectbox("語言(Language)", ["繁體中文", "English"])
            st.session_state["chat_lang"] = chat_lang
    return car_model, chat_lang

def display_chat_messages():
    """
    Display chat messages and uploaded images in the Streamlit app.
    顯示聊天消息和上傳的圖片。
    """
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                display_user_message(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                display_assistant_message(message.content)

def display_user_message(message_content: Union[str, List[dict]]) -> None:
    """
    Display user message in the chat message.
    顯示用戶消息。
    """
    if isinstance(message_content, str):
        message_text = message_content
    elif isinstance(message_content, dict):
        message_text = message_content["input"][0]["content"][0]["text"]
    else:
        message_text = message_content[0]["text"]

    message_content_markdown = message_text.split('</context>\n\n', 1)[-1] if message_text is not None else ""
    st.markdown(message_content_markdown)


def display_assistant_message(message_content: Union[str, dict]) -> None:
    """
    Display assistant message in the chat message.
    顯示助理消息。
    """
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])

def langchain_messages_format(
    messages: List[Union["AIMessage", "HumanMessage"]]
) -> List[Union["AIMessage", "HumanMessage"]]:
    """
    Format the messages for the LangChain conversation chain.
    格式化 LangChain 對話鏈的消息。
    """
    for i, message in enumerate(messages):
        if isinstance(message.content, list):
            if "role" in message.content[0]:
                if message.type == "ai":
                    message = AIMessage(message.content[0]["content"])
                if message.type == "human":
                    message = HumanMessage(message.content[0]["content"])
                messages[i] = message
    return messages