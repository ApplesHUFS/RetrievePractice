import json
import torch
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import argparse
import os

def load_data(file_path):
    print(f"데이터 로드 중: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"데이터 로드 완료: {len(data)}개 문서")
    return data

def add_metadata(documents):
    print("메타데이터 추가 중...")
    for doc in documents:
        doc['metadata'] = {
            'title_length': len(doc['title']),
            'content_length': len(doc['content']),
            'total_length': len(doc['title']) + len(doc['content']),
            'title_word_count': len(doc['title'].split()),
            'content_word_count': len(doc['content'].split()),
            'combined_text': f"{doc['title']} {doc['content']}"
        }
    print("메타데이터 추가 완료")
    return documents

def create_bm25_preprocessed(documents):
    print("BM25 전처리 중...")
    for doc in documents:
        doc['bm25_tokens'] = doc['metadata']['combined_text'].split()
    print("BM25 전처리 완료")
    return documents

def create_dpr_embeddings(documents):
    print("DPR 임베딩 생성 중...")
    
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    for i, doc in enumerate(documents):
        print(f"문서 {i+1}/{len(documents)} 인코딩 중...")
        
        inputs = context_tokenizer(
            doc['metadata']['combined_text'], 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = context_encoder(**inputs)
        
        embedding = outputs.pooler_output.detach().numpy()
        doc['dpr_embedding'] = embedding.tolist()[0] 
        
    print("DPR 임베딩 생성 완료")
    return documents

def create_sentence_transformer_embeddings(documents):
    print("Sentence Transformer 임베딩 생성 중...")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    texts = [doc['metadata']['combined_text'] for doc in documents]
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc['st_embedding'] = embedding.tolist()
        
    print("Sentence Transformer 임베딩 생성 완료")
    return documents

def save_preprocessed_data(documents, output_file):
    print(f"전처리된 데이터 저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print("전처리된 데이터 저장 완료")
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"저장된 파일 크기: {file_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='RAG 전처리 도구')
    parser.add_argument('--input', type=str, default='data/documents.json', help='입력 JSON 파일 경로')
    parser.add_argument('--output', type=str, default='data/preprocessed_documents.json', help='출력 JSON 파일 경로')
    args = parser.parse_args()
    
    documents = load_data(args.input)
    
    documents = add_metadata(documents)
    documents = create_bm25_preprocessed(documents)
    documents = create_dpr_embeddings(documents)
    documents = create_sentence_transformer_embeddings(documents)
    
    save_preprocessed_data(documents, args.output)
    
    print("모든 전처리 작업 완료")

if __name__ == "__main__":
    main()
