import os
from dotenv import load_dotenv
import openai
import streamlit as st
from langchain.document_loaders import PDFMinerLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
FAISS_DB_DIR = "vectorstore"


def _modify_newlines(text):
    """
    テキストの前処理関数.
    単一の\nは削除し, 2連続の改行(\n\n)は\nに変更する.
    """
    temporary_string = '¡¡'
    text = text.replace('\n\n', temporary_string)
    text = text.replace('\n', '')
    text = text.replace(temporary_string, '\n')
    return text


@st.cache_resource
def preprocess():
    """
    RAGの前処理.
    """

    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_DB_DIR):
        print("load vector store")
        _faiss_db = FAISS.load_local(FAISS_DB_DIR, embeddings=embeddings)
    else:
        print("start preprocess")
        # pdfファイルの読み込み
        loader = PDFMinerLoader("./document/ms_ad/car_insurance.pdf")
        raw_documents = loader.load()

        # チャンク分け
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)

        #  テキストの前処理
        for document in documents:
            document.page_content = _modify_newlines(document.page_content).replace(" ", "").replace("　", "")

        # ベクトルストアの構築
        _faiss_db = FAISS.from_documents(documents=documents, embedding=embeddings)
        _faiss_db.save_local(FAISS_DB_DIR)
        print("finish preprocess")

    # モデル
    model = ChatOpenAI(model="gpt-4", temperature=0, client=openai.ChatCompletion)

    # QAモデル
    _qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=_faiss_db.as_retriever())
    return _faiss_db, _qa


# 前処理
faiss_db, qa = preprocess()

st.title("三井住友海上 保険チャットボット")

# メッセージ履歴を保持するリストを初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# メッセージ履歴の表示
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("三井住友海上の車の保険に関してご質問ください！"):

    # 質問の保存・保存
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    query = f"三井住友海上の車の保険についての質問に答えるチャットボットです。お客様に向けて次の質問に丁寧に答えてください。:{prompt}"
    with get_openai_callback() as cb:
        response = qa.run(query)

    # 回答の表示・保存
    st.chat_message("assistant").markdown(response)
    print(f"[RESPONSE]: {response}")
    print(f"{cb.total_cost}")
    st.session_state.messages.append({"role": "assistant", "content": response})
