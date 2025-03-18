import json
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from sentence_transformers import SentenceTransformer, util
import time

class BaseRetriever:
    def __init__(self, data_path):
        self.load_data(data_path)
    
    def load_data(self, data_path):
        print(f"데이터 로드 중: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        print(f"데이터 로드 완료: {len(self.documents)}개 문서")
    
    def retrieve(self, query, top_k=3):
        raise NotImplementedError("이 메서드는 상속 클래스에서 구현해야 합니다.")
    
    def format_results(self, results, query, elapsed_time):
        return {
            "query": query,
            "retriever": self.__class__.__name__,
            "elapsed_time": elapsed_time,
            "results": results
        }

class BM25Retriever(BaseRetriever):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.setup_bm25()
    
    def setup_bm25(self):
        print("BM25 인덱스 설정 중...")
        
        tokenized_corpus = [doc['bm25_tokens'] for doc in self.documents]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("BM25 인덱스 설정 완료")
    
    def retrieve(self, query, top_k=3):
        print(f"BM25 검색 수행 중... 쿼리: '{query}'")
        start_time = time.time()
        
        tokenized_query = query.split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "rank": i+1,
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "score": float(scores[idx]),
                "content": self.documents[idx]["content"]
            })
        
        elapsed_time = time.time() - start_time
        print(f"BM25 검색 완료: {elapsed_time:.4f}초")
        
        return self.format_results(results, query, elapsed_time)

class DPRRetriever(BaseRetriever):
    
    def __init__(self, data_path):
        super().__init__(data_path)
        self.setup_dpr()
    
    def setup_dpr(self):
        """DPR 모델 및 임베딩 설정 함수"""
        print("DPR 모델 설정 중...")
        
        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        
        self.document_embeddings = np.array([doc['dpr_embedding'] for doc in self.documents])
        
        print("DPR 모델 설정 완료")
    
    def retrieve(self, query, top_k=3):
        print(f"DPR 검색 수행 중... 쿼리: '{query}'")
        start_time = time.time()
        
        inputs = self.question_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
        query_embedding = outputs.pooler_output.detach().numpy()
        
        scores = np.dot(self.document_embeddings, query_embedding.T).squeeze()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "rank": i+1,
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "score": float(scores[idx]),
                "content": self.documents[idx]["content"]
            })
        
        elapsed_time = time.time() - start_time
        print(f"DPR 검색 완료: {elapsed_time:.4f}초")
        
        return self.format_results(results, query, elapsed_time)

class SentenceTransformerRetriever(BaseRetriever):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.setup_sentence_transformer()
    
    def setup_sentence_transformer(self):
        print("Sentence Transformer 모델 설정 중...")
        
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.document_embeddings = torch.tensor([doc['st_embedding'] for doc in self.documents])
        
        print("Sentence Transformer 모델 설정 완료")
    
    def retrieve(self, query, top_k=3):
        print(f"Sentence Transformer 검색 수행 중... 쿼리: '{query}'")
        start_time = time.time()
        
        query_embedding = self.model.encode(query)
        query_embedding_tensor = torch.tensor(query_embedding)
        
        cos_scores = util.cos_sim(query_embedding_tensor, self.document_embeddings)[0]
        
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.documents)))
        top_indices = top_results[1].tolist()
        top_scores = top_results[0].tolist()
        
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            results.append({
                "rank": i+1,
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "score": float(score),
                "content": self.documents[idx]["content"]
            })
        
        elapsed_time = time.time() - start_time
        print(f"Sentence Transformer 검색 완료: {elapsed_time:.4f}초")
        
        return self.format_results(results, query, elapsed_time)
