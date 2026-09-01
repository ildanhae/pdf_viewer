
import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드 (OPENAI_API_KEY)
load_dotenv()

st.set_page_config(page_title="PDF RAG 챗봇", page_icon="📚")
st.title("📚 Multi-PDF 문서 기반 질의응답 시스템")

# 1. 파일 업로드 UI (여러 PDF 지원)
uploaded_files = st.sidebar.file_uploader(
    "PDF 파일을 업로드하세요 (복수 선택 가능)", 
    type=["pdf"], 
    accept_multiple_files=True
)

# PDF 텍스트 추출 함수
def extract_text_from_pdfs(files):
    combined_text = ""
    for file in files:
        # Streamlit UploadedFile 객체를 PyMuPDF로 읽기
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            combined_text += page.get_text() + "\n\n"
    return combined_text

# FAISS 벡터스토어 생성 및 캐싱 (동일 파일에 대해 중복 계산 방지)
@st.cache_resource(show_spinner="문서를 분석하고 벡터 DB를 생성하는 중입니다...")
def create_vectorstore(_files):
    raw_text = extract_text_from_pdfs(_files)

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(raw_text)

    # 임베딩 및 FAISS 벡터스토어 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# 문서 분석 및 질의응답 진행
if uploaded_files:
    # 벡터스토어 생성
    vectorstore = create_vectorstore(uploaded_files)
    st.sidebar.success(f"{len(uploaded_files)}개의 PDF 분석 완료!")

    # 사용자 질문 입력
    query = st.text_input("질문을 입력하세요:", placeholder="예: 매달 내야하는 보험료 알려줘.")

    if query:
        with st.spinner("답변을 생성하는 중입니다..."):
            # 유사도 검색 (상위 10개 청크 extraction)
            search_results = vectorstore.similarity_search_with_score(query, k=10)

            # 검색된 컨텍스트 결합
            context = "\n\n".join([doc[0].page_content for doc in search_results])

            # 프롬프트 및 체인 설정
            prompt = ChatPromptTemplate.from_template("""다음 배경지식을 사용해서 사용자 질문에 대답해.

[배경 지식]
{context}

[사용자 질문]
{question}""")

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            chain = prompt | llm | StrOutputParser()

            # 답변 실행
            response = chain.invoke({"context": context, "question": query})

            # 결과 출력
            st.markdown("### 💬 답변")
            st.write(response)

            # 참고한 문서 구간 접기/펴기 기능 (검증용)
            with st.expander("🔍 참조한 문서 내용 보기"):
                for i, doc in enumerate(search_results, 1):
                    st.markdown(f"**Chunk {i} (Similarity Score: {doc[1]:.4f})**")
                    st.text(doc[0].page_content)
                    st.divider()
else:
    st.info("왼쪽 사이드바에서 PDF 파일을 하나 이상 업로드해주세요.")
