from positional_inverted_index import PositionalInvertedIndex
from english_preprocessing import startPipeline as startEnglishPipeline
from arabic_preprocessing import startPipeline as startArabicPipeline
from pathlib import Path
from time import time

class SearchEngine(PositionalInvertedIndex):
    def __init__(self) -> None:
        start = time()
        projectPath = Path(__file__).parent
        startEnglishPipeline(projectPath / "data" / "en", projectPath / "Intermediate" /"Processed Documents" / "en" )
        startArabicPipeline(projectPath / "data" / "ar", projectPath / "Intermediate" /"Processed Documents" / "ar" )
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

def test(engine: SearchEngine) -> None:
    testingExamples = {
        "artificial intelligence": ["Artificial_intelligence.txt", "Deep_learning.txt", 'en_001.txt'],
        "deep learning neural networks": [ "Deep_learning.txt", "Artificial_intelligence.txt", 'en_001.txt'],
        "random access memory": ["Random-access_memory.txt"],
        "firewalls encryption": ["Computer_security.txt", 'en_004.txt'],
        "cloud computing edge computing": ['en_002.txt'],
        "data warehousing": ['en_003.txt'],
        "javascript web browsers": ["JavaScript.txt"],
        "unix bell labs": ["Unix.txt"],
        "هتلر": ["الحرب_العالمية_الثانية.txt"],
        "التعلم العميق": ["تعلم_عميق.txt", "ذكاء_اصطناعي.txt", 'ar_001.txt'],
        "المصادقة متعددة العوامل": ["ar_004.txt"],
        "الجدران النارية": ["جدار_حماية_(حوسبة).txt", "أمن_الحاسوب.txt", 'ar_004.txt']
    }
    
    for query, relevantDocs in testingExamples.items():
        print(f"Testing Query: '{query}'")
        
        results = engine.search(query)
        retrieved_docs = []
        
        for docId, _ in results:
            doc_name = engine.docIdMap.get(docId, "Unknown")
            retrieved_docs.append(doc_name)
            
        print(f"Retrieved {len(retrieved_docs)} docs: {retrieved_docs}")
        print(f"Relevant Expected: {relevantDocs}")
        
        retrievedSet = set(retrieved_docs)
        relevantSet = set(relevantDocs)
        
        truePositive = len(retrievedSet.intersection(relevantSet))
        
        if len(retrievedSet) > 0:
            precision = truePositive / len(retrievedSet)
        else:
            precision = 0.0
        
        if len(relevantSet) > 0: 
            recall = truePositive / len(relevantSet)
        else:
            recall = 0.0
    
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print("====================================")
        
def main() -> None:
    engine = SearchEngine()
    test(engine)
    run_app(engine)

if __name__ == "__main__":
    main()
