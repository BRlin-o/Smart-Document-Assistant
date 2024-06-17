from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

rag_prompt = """只能使用以下內容來回答最後的問題，並遵守以下規則：
1. 如果不知道答案，請不要編造答案。只需說 **找不到最終答案**。
2. 如果找到答案，請務必詳細地正確回答，並附上**直接**用來得出答案的來源列表。排除與最終答案無關的來源。
3. 如果內容包含 Markdown 格式的 ![]()，請在回應中保留它完整的樣貌（一模一樣）。
4. 完全使用繁體中文進行溝通。

{context}

問題：{question}
有幫助的回答：""" ## score最高的

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
vbd_path = "embed_documents/ALL"
vdb = FAISS.load_local(folder_path=vbd_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def get_rag_chain(llm, car_model=None, retriever_k=4):
    retriever = vdb.as_retriever(search_kwargs={"k": retriever_k, 'filter': {"car_model": car_model}})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=rag_prompt,
                input_variables=["context", "question"],
            ),
        },
        retriever=retriever,
        return_source_documents=True,
        input_key="question",
    )
    return qa