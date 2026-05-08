from positional_inverted_index import PositionalInvertedIndex
from english_preprocessing import startPipeline as startEnglishPipeline
from arabic_preprocessing import startPipeline as startArabicPipeline
from pathlib import Path
from time import time

class SearchEngine(PositionalInvertedIndex):
    def __init__(self) -> None:
        start = time()
        project_path = Path(__file__).parent
        startEnglishPipeline(project_path / "data" / "en", project_path / "Intermediate" /"Processed Documents" / "en" )
        startArabicPipeline(project_path / "data" / "ar", project_path / "Intermediate" /"Processed Documents" / "ar" )
        super().__init__()
        print(f"took {time() - start}s to start Engine")


def run_app(engine: SearchEngine) -> None:
    print("Enter a query to search")
    print("Enter '' to exit...")
    print("Use proximity syntax like: machine /3 learning")

    while True:
        query = input("\nSearch query: ").strip()
        if query == "" :
            break
        if "/" in query:
            engine.kSearch(query)
        else:
            engine.search(query)


def main() -> None:
    test()
    engine = SearchEngine()
    run_app(engine)

def test() -> None:
    engine = SearchEngine()
    ground_truth = {
        "artificial intelligence": ["Artificial_intelligence.txt", "Deep_learning.txt"],
        "computer memory": ["Random-access_memory.txt", "Computer_hardware.txt"],
        "نظام التشغيل": ["نظام_التشغيل_لينكس.txt", "نظام_التشغيل.txt"],
        "هتلر": ["الحرب_العالمية_الثانية.txt"], # <----- اهم حاجة دا 
        "كمبيوتر": ["امن_الحاسوب.txt", "وحدة_معالجة_مركزية.text", "جدار_حماية_(حوسبة).txt"]
    }
    
    for query, relevant_docs in ground_truth.items():
        print(f"Testing Query: '{query}'")
        
        results = engine.search(query)
        retrieved_docs = []
        
        for docId, _ in results:
            doc_name = engine.docIdMap.get(docId, "Unknown")
            retrieved_docs.append(doc_name)
            
        print(f"Retrieved {len(retrieved_docs)} docs: {retrieved_docs}")
        print(f"Relevant Expected: {relevant_docs}")
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        true_positive = len(retrieved_set.intersection(relevant_set))
        
        if len(retrieved_set) > 0:
            precision = true_positive / len(retrieved_set)
        else:
            precision = 0.0
        
        if len(relevant_set) > 0: 
            recall = true_positive / len(relevant_set)
        else:
            recall = 0.0
    
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print("====================================")

if __name__ == "__main__":
    main()
