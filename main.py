import json
import argparse
from retriever import BM25Retriever, DPRRetriever, SentenceTransformerRetriever

def print_results(results):
    print("\n" + "="*80)
    print(f"검색 쿼리: '{results['query']}'")
    print(f"검색 방법: {results['retriever']}")
    print(f"검색 소요 시간: {results['elapsed_time']:.4f}초")
    print("-"*80)
    
    for result in results['results']:
        print(f"[{result['rank']}] ID: {result['id']}, 점수: {result['score']:.4f}")
        print(f"제목: {result['title']}")
        print(f"내용: {result['content'][:150]}..." if len(result['content']) > 150 else f"내용: {result['content']}")
        print("-"*80)

def main():
    parser = argparse.ArgumentParser(description='RAG 검색 시스템')
    parser.add_argument('--data', type=str, default='data/preprocessed_documents.json', help='전처리된 데이터 파일 경로')
    args = parser.parse_args()
    
    print("검색기 초기화 중...")
    retrievers = {
        '1': ('BM25 (키워드 기반)', BM25Retriever(args.data)),
        '2': ('DPR (Dense Passage Retrieval)', DPRRetriever(args.data)),
        '3': ('Sentence Transformer', SentenceTransformerRetriever(args.data))
    }
    print("검색기 초기화 완료")
    
    while True:
        print("\n" + "="*80)
        print("RAG 검색 시스템")
        print("="*80)
        
        query = input("\n검색어를 입력하세요 (종료: q): ")
        if query.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        
        print("\n검색 방법을 선택하세요:")
        for key, (name, _) in retrievers.items():
            print(f"{key}. {name}")
        print("4. 모든 방법으로 검색")
        
        choice = input("선택 (1-4): ")
        
        if choice in retrievers:
            name, retriever = retrievers[choice]
            results = retriever.retrieve(query)
            print_results(results)
        elif choice == '4':
            all_results = []
            for name, retriever in retrievers.values():
                results = retriever.retrieve(query)
                print_results(results)
                all_results.append(results)
            
            print("\n" + "="*80)
            print("검색 방법 비교")
            print("-"*80)
            
            print("각 방법별 상위 문서 ID:")
            for i, results in enumerate(all_results):
                method_name = results['retriever']
                doc_ids = [r['id'] for r in results['results']]
                print(f"{i+1}. {method_name}: {doc_ids}")
        else:
            print("잘못된 선택입니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()
