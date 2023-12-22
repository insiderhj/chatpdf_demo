from typing import Any
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader

import fitz
from PIL import Image

import chromadb
import re
import uuid

css = """
.testtest {width: 50px !important}
"""

# OpenAI API 키 입력 상자와 비활성화 상자 정의
enable_box = gr.Textbox.update(value=None, placeholder='Upload your OpenAI API key', interactive=True)
disable_box = gr.Textbox.update(value='OpenAI API key is Set', interactive=False)

# OpenAI API 키 설정 함수
def set_apikey(api_key):
    if not api_key:
        api_key = "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"

    app.OPENAI_API_KEY = api_key

    return disable_box

# OpenAI API 키 입력 상자 활성화 함수
def enable_api_box():
    return enable_box

# 텍스트 추가 함수
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY=None) -> None:
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.chain = None
        self.chat_history = []
        self.doc_source = ''
        self.doc_page = 0
        self.count = 0

    def __call__(self, files) -> Any:
        if self.count == 0:
            print('This is here')
            self.chain = self.build_chain(files)  # self.chain에 반환 값을 할당합니다.
            self.count += 1

        return self.chain  # self.chain을 반환합니다.

    # Chroma 데이터베이스 클라이언트 생성
    def chroma_client(self):
        client = chromadb.Client()
        collection = client.get_or_create_collection(name="my-collection")
        return client

    # PDF 파일 처리 함수
    def process_file(self, files):
        documents = []
        for file in files:
            loader = PyPDFLoader(file.name)
            documents += loader.load()
        pattern = r"/([^/]+)$"
        match = re.search(pattern, files[0].name)
        file_name = match.group(1)
        return documents, file_name

    # # PDF 파일 처리 함수
    # def process_file(self, file):
    #     loader = PyPDFLoader(file.name)
    #     documents = loader.load()
    #     pattern = r"/([^/]+)$"
    #     match = re.search(pattern, file.name)
    #     file_name = match.group(1).replace(".pdf", "").replace(" ", "_")  # 확장자를 제거하고 공백을 밑줄로 대체합니다.
    #     # 그 외의 특수문자도 필요에 따라 제거하거나 대체할 수 있습니다.
    #     return documents, file_name


    # 대화형 검색 체인 구축 함수
    def build_chain(self, files):
        documents, file_name = self.process_file(files)
        # OpenAI Embeddings 모델 로드
        default_api_key = "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"  # 기본 키 값을 설정합니다.
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY or default_api_key)
        pdfsearch = Chroma.from_documents(
            documents,
            embeddings,
            collection_name=file_name,
        )

        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY or default_api_key),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
        )
        return chain

# 응답 처리 함수
def get_response(history, query, files):
    if not files:
        raise gr.Error(message='Upload Files')

    chain = app(files)

    result = chain({"question": query, 'chat_history': app.chat_history}, return_only_outputs=True)
    app.chat_history += [(query, result["answer"])]
    app.doc_source = list(result['source_documents'][0])[1][1]['source']
    app.doc_page = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''

def move_left():
    app.doc_page = 0


def move_right():
    app.doc_page = 0

# 파일 렌더링 함수
def render_file(files):
    for file in files:
        if file.name == app.doc_source:
            doc = fitz.open(file.name)
            print(file.name)
    page = doc[app.doc_page]
    # 300 DPI의 해상도로 페이지를 PNG 이미지로 렌더링합니다.
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

# 첫 번째 페이지 렌더링 함수
def render_first(files):
    doc = fitz.open(files[0].name)
    app.doc_page = 0
    page = doc[0][app.doc_page]
    # 300 DPI의 해상도로 페이지를 PNG 이미지로 렌더링합니다.
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image, []

app = my_app()
with gr.Blocks() as demo:
    state = gr.State(uuid.uuid4().hex)
    with gr.Row():
        chatbot = gr.Chatbot(value=[], elem_id='chatbot')
        left_btn = gr.Button("<")
        show_img = gr.Image(tool='select')
        right_btn = gr.Button(">")
    with gr.Row():
        with gr.Column(scale=6):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            )
        with gr.Column(scale=2):
            submit_btn = gr.Button('Submit')
        with gr.Column(scale=2):
            btn = gr.UploadButton("📁 Upload Files", file_count="multiple", file_types=[".pdf"])


    # api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])
    # change_api_key.click(fn=enable_api_box, outputs=[api_key])
    btn.upload(fn=render_first, inputs=[btn], outputs=[show_img, chatbot])

    submit_btn.click(fn=add_text, inputs=[chatbot, txt], outputs=[chatbot], queue=False).success(
        fn=get_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

    left_btn.click(fn=move_left).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

    right_btn.click(fn=move_right).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

demo.queue()
demo.launch(share=True)
