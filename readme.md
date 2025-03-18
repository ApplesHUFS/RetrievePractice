# RAG 검색 방법 비교 구현

이 프로젝트는 Retrieval-Augmented Generation(RAG)의 검색(Retrieval) 단계를 3가지 방법으로 구현하고 비교합니다.

## 구현된 검색 방법

1. **BM25** - 전통적인 키워드 기반 검색 알고리즘
2. **DPR(Dense Passage Retrieval)** - 질의와 문서를 각각 다른 인코더로 임베딩하는 방식
3. **Sentence Transformer** - 문장/문서를 의미적 벡터 공간으로 변환하는 방식

## 프로젝트 구조

- `preprocess.py`: 데이터 전처리 및 임베딩 생성
- `retriever.py`: 검색 모듈 (BM25, DPR, Sentence Transformer)
- `main.py`: 메인 실행 모듈
- `requirements.txt`: 필요한 패키지 목록

## 실행 방법

### 1. 필요 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리

```bash
python preprocess.py --input documents.json --output preprocessed_documents.json
```

### 3. 검색 실행

```bash
python main.py --data preprocessed_documents.json
```

## 전처리 과정

전처리 과정에서는 다음 작업이 수행됩니다:

1. 메타데이터 추가
   - 문서 길이, 단어 수 등 기본 정보
   - 제목과 내용을 결합한 텍스트

2. BM25 전처리
   - 토큰화된 형태로 저장

3. DPR 임베딩 생성
   - DPR Context Encoder로 문서 임베딩 생성

4. Sentence Transformer 임베딩 생성
   - Sentence Transformer로 문서 임베딩 생성

## 데이터 형식

데이터는 다음과 같은 형식의 JSON 파일이어야 합니다:

```json
[
  {
    "id": 1,
    "title": "인공지능의 역사",
    "content": "인공지능(AI)의 역사는 1950년대부터 시작됩니다. 앨런 튜링은 1950년에 '기계가 생각할 수 있는가?'라는 질문을 담은 논문을 발표했습니다. 이후 1956년 다트머스 회의에서 '인공지능'이라는 용어가 처음 사용되었습니다."
  },
  {
    "id": 2,
    "title": "딥러닝의 발전",
    "content": "딥러닝은 2010년대부터 급속도로 발전했습니다. 2012년 AlexNet이 ImageNet 대회에서 우승하면서 딥러닝 열풍이 시작되었고, 이후 이미지 인식, 자연어 처리 등 다양한 분야에서 혁신을 가져왔습니다."
  }
]
```

각 문서는 최소한 `id`, `title`, `content` 필드를 포함해야 합니다.

## 검색 과정

검색 시스템은 다음과 같은 절차로 동작합니다:

1. 사용자 쿼리 입력
2. 검색 방법 선택 (BM25, DPR, Sentence Transformer, 전체 비교)
3. 선택한 방법으로 검색 수행
4. 결과 출력 및 비교
