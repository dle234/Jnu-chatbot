# 전남대학교 공지사항 크롤러
import re
import time
import requests
from io import BytesIO
from PIL import Image
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os
from PIL import Image

# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'
chrome_options = Options()
chrome_options.add_argument("--headless") 
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 유효한 카테고리 정의
valid_categories = ["학사안내", "대학생활", "모집공고", "공모전", "채용공고", "취업정보", "장학안내", "병무안내"]

# 크롤링된 데이터 저장 리스트
data = []

# 포털 기본 주소
base_url = "https://www.jnu.ac.kr"

# 페이지 범위 설정
for page in range(1, 2):
    print(f"📄 [페이지 {page}] 크롤링 중...")
    url = f"https://www.jnu.ac.kr/WebApp/web/HOM/COM/Board/board.aspx?boardID=5&bbsMode=list&cate=0&page={page}"
    driver.get(url)
    time.sleep(2)
    
    # 전남대 홈페이지 공지사항에서 제목 태그
    notice_elements = driver.find_elements(By.CSS_SELECTOR, "td.title > a")
    
    # 홈페이지 제목수집
    notices_to_process = []
    for el in notice_elements:
        try:
            full_title = el.text.strip()
            relative_link = el.get_attribute("href")
            link = relative_link 
            match = re.match(r"\[(.*?)\]\s*(.*)", full_title)
            # 카테고리 별 분류
            if match:
                category = match.group(1)
                title = match.group(2)
            else:
                category = "기타"
                title = full_title
                
            if category not in valid_categories:
                category = "기타"
                
            notices_to_process.append({
                "category": category,
                "title": title,
                "link": link
            })
        except Exception as e:
            print(f"⚠️ 제목/링크 추출 실패: {e}")
    
    # 수집된 링크를 하나씩 방문하여 내용 크롤링
    for notice in notices_to_process:
        try:
            print(f"🔍 처리 중: [{notice['category']}] {notice['title']}")
            driver.get(notice['link'])
            time.sleep(2)
            
            # 본문 텍스트 수집
            try:
                text_content = driver.find_element(By.CLASS_NAME, "view_body").text.strip()
                text_content = text_content.replace("\n", " ").replace("\r", " ")
            except Exception as e:
                print(f"⚠️ 본문 텍스트 추출 실패: {e}")
                text_content = ""
            
            # 이미지 OCR 텍스트 수집
            ocr_text = ""
            try:
                images = driver.find_elements(By.CSS_SELECTOR, ".view_body img")
                for img in images:
                    img_url = img.get_attribute("src")
                    if not img_url.startswith("http"):
                        img_url = base_url + img_url
                    
                    try:
                        response = requests.get(img_url)
                        image = Image.open(BytesIO(response.content))
                        
                        ocr_text += pytesseract.image_to_string(image, lang="kor") + "\n"
                    except Exception as e:
                        print(f"⚠️ 이미지 OCR 실패: {e}")
            except Exception as e:
                print(f"⚠️ 이미지 요소 찾기 실패: {e}")
            
            # 수집한 내용 저장
            content = text_content + "\n" + ocr_text.strip()
            notice["content"] = content
            data.append(notice)
            
        except Exception as e:
            print(f"⚠️ 공지 크롤링 실패: {e}")
driver.quit()

# ✅ 결과 확인 
print(f"\n✅ 총 {len(data)}개의 공지 수집 완료!\n")
for notice in data[:3]:
    print(f"[{notice['category']}] {notice['title']}")
    print(f"🔗 {notice['link']}")
    print(notice['content'][:100] + "..." if notice['content'] else "내용 없음", "\n")
    
    
    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# langchain을 활용하여 데이터 청킹 
def chunk_documents(data, chunk_size=150, chunk_overlap=25):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_documents = []

    for item in data:
        metadata = {
            "category": item.get("category", ""),
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "source_id": item.get("id", "") if "id" in item else f"{item.get('title', '')[:20]}"
        }

        title = item.get("title", "")
        category = item.get("category", "")
        content = item.get("content", "")
        link = item.get("link", "")

        if content:
            content_chunks = text_splitter.split_text(content)

            for i, chunk in enumerate(content_chunks):
                # 방식 1번 : full_chunk = chunk
                # 방식 2번 : category + title + chunk 조합
                full_chunk = f"[{category}] {title} - {chunk}"
                # full_chunk += f" (출처: {link})"
                
                doc = Document(
                    page_content=full_chunk,
                    metadata={
                        **metadata,
                        "chunk_id": i,
                        "chunk_count": len(content_chunks)
                    }
                )
                
                chunked_documents.append(doc)
        else:
            full_chunk = f"[{category}] {title}"
            doc = Document(
                page_content=full_chunk,
                metadata=metadata
            )
            chunked_documents.append(doc)

    return chunked_documents



# 한국어 전용 SBERT 모델 로딩
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 예시 임베딩 함수
def embed_documents(docs):
    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return embeddings

chunked_docs = chunk_documents(data)
embed_documents(chunked_docs)

# Pinecone 디비 업로드

from pinecone import Pinecone, ServerlessSpec
import numpy as np
import json

# Pinecone 클라이언트 초기화
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
INDEX_NAME = "jnu-notice"

# 청킹한 데이터 임베딩 후 Pinecone 업로드
def upload_documents(docs, index_name=INDEX_NAME):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    
    if index_name in pc.list_indexes().names():
        print(f"⚠️ {index_name} 인덱스가 이미 존재합니다. 삭제 후 재생성합니다.")
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    texts = [doc.page_content for doc in docs]
    
    # 임베딩
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # 임베드 된 정보와 해당 내용 같이 업로드
    vectors = [
        (f"id_{i}", emb.tolist(), {"text": texts[i]})
        for i, emb in enumerate(embeddings)
    ]

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])
        print(f"✅ {i+len(vectors[i:i+100])}개 업로드됨")

    return index

uploaded_index = upload_documents(chunked_docs, index_name=INDEX_NAME)

print("✅ Pinecone 업로드 완료!",uploaded_index)

import google.generativeai as genai
from tqdm import tqdm
# 언어모델 연결

import streamlit as st

# Streamlit secrets에서 API 키 불러오기
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Gemini 설정
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# Gemini 질문하기
def query_with_gemini(index, query, top_k=10):
    try:
        # 임베딩
        query_embedding = embedding_model.encode(query).tolist()
        
        # Pinecone 검색
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        matches = search_results.get("matches", [])
        if not matches:
            return "🔍 검색 결과가 없습니다. 질문이 너무 구체적이거나 관련 문서가 없을 수 있어요."
        # 검색된 문서 추출
        retrieved_chunks = [
            match['metadata']['text']
            for match in matches
            if match.get('metadata') and match['metadata'].get('text')
        ]
        
        # 최대 길이 제한 (Gemini 입력 길이 방지)
        context_text = "\n\n".join(retrieved_chunks)[:10000]

        # 프롬프트 구성
        prompt = f"""
🔎 [문서 검색 결과 요약]:
다음은 '{query}'에 대해 검색된 문서 조각들입니다.

{context_text}

🧠 [당신의 역할]:
- 당신은 지식 기반의 정보를 바탕으로 정확하고 신뢰성 있는 답변을 제공하는 전문가입니다.

📌 [답변 지침]:
1. 문서에 관련 내용이 **명확히 포함되어 있다면**, 그 내용을 인용해 답변하세요.
2. **직접적인 정보가 없더라도**, 문서 맥락을 분석하여 가능한 **추론/해석**을 시도해 주세요.
3. 답변은 정확하되, **친절하고 자연스러운 한국어**로 설명해 주세요.
4. 답변과 전혀 관련 없는 내용은 생략해주세요
5. "제공된 자료에는*, 문서에는* " 으로 시작하지 말아줘.
6. 너무 딱딱하지 않게 , 친절하고 대화하듯 "*요"로 마무리 해줘.
7. 가독성이 좋게 각 문장들을 줄바꿈 해줘. 
8. 문서에 적절한 이모티콘을 넣어 읽기 쉽게 만들어줘.

❓ [질문]:
{query}

📝 [답변]:
"""
        # Gemini 응답 생성
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=0.8
            )
        )

        return response.text.strip()
    
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}"


# ✅ 실제 실행
# query = "취업 관련 공고가 있나?"
# query = "졸업 유보를 하고싶은데, 유보의 조건"
query = "휴학을 하고 싶어. 휴학 신청 기간 알려줘."

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
answer = query_with_gemini(uploaded_index, query)

print("\n📢 Gemini 응답:")
print(answer)
