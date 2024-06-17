import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain.prompts.prompt import PromptTemplate
# from langchain.callbacks.manager import CallbackManager
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains.question_answering import load_qa_chain
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

QA_PROMPT = """只能使用以下內容來回答最後的問題，並遵守以下規則：
1. 如果不知道答案，請不要編造答案。只需說 **找不到最終答案**。
2. 如果找到答案，請務必詳細地正確回答，並附上**直接**用來得出答案的來源列表。排除與最終答案無關的來源。
3. 如果score最高的內容包含 Markdown 格式的 ![]()，請在回應中保留它完整的樣貌（一模一樣）。
4. 完全使用繁體中文進行溝通。

{context}

問題：{question}
有幫助的回答："""

car_model_list = ["CrossOver", "Delight", "SuperSport"]
car_model = "DELIGHT"
retriever_k = 5

st_callback = StreamlitCallbackHandler(st.container())
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

llm = ChatNVIDIA(
    model="meta/llama3-70b-instruct", 
    # model="nvidia/nemotron-4-340b-instruct", 
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"), 
    temperature=0.0,
    top_p=1,
    max_tokens=1024,
    streaming=True,
    verbose=True,
    callbacks=[StreamHandler(st.empty())]
)

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
embedding_path = "embed_documents/ALL"

# tools = load_tools([])
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )
db = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def set_CAR_MODEL_From_Session():
    pass
    global CAR_MODEL
    CAR_MODEL = st.session_state["car_model"]
    print("[DEBUG] set_CAR_MODEL_From_Session", CAR_MODEL)

st.title("🏍️ Gogoro Smart Scooter 萬事通")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        car_model = st.selectbox(
            "車型", 
            car_model_list,
            on_change=set_CAR_MODEL_From_Session
        )
        car_model = car_model.upper()
        st.session_state["car_model"] = car_model
    with col2:
        retriever_k = st.slider(
            "Retriever Top K",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
        )

retriever = db.as_retriever(search_kwargs={"k": retriever_k, 'filter': {"car_model": car_model}})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "question"],
        ),
    },
    retriever=retriever,
    return_source_documents=True,
    input_key="question",
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(
            qa.stream(
                {
                    "question": prompt
                }, 
                # {
                #     "callbacks": [StreamHandler(st.empty())]
                # }
            )
        )
# messages = [
#     SystemMessage(content="你是一个能干的助手."),
#     HumanMessage(content="谁赢得了2020年的世界职业棒球大赛?"),
#     AIMessage(content="洛杉矶道奇队在2020年赢得了世界职业棒球大赛冠军."),
#     HumanMessage(content="它在哪里举办的?")
# ]

# print(llm.invoke(messages))
