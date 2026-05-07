from positional_inverted_index import PositionalInvertedIndex
import time

def evaluate():
    print("Loading Index...")
    start_time = time.time()
    idx = PositionalInvertedIndex()
    end_time = time.time()
    print(f"Index loaded in {end_time - start_time:.4f} seconds.\n")

    # Define test queries and the documents you expect them to return
    # IMPORTANT: Update these keys to your sample queries and 
    # update the lists to exactly match your document filenames!
    ground_truth = {
        "artificial intelligence": ["Artificial_intelligence.txt", "Deep_learning.txt"],
        "computer memory": ["Random-access_memory.txt", "Computer_hardware.txt"],
        "نظام التشغيل": ["نظام_التشغيل_لينكس.txt", "نظام_التشغيل.txt"] # Replace with your actual Arabic filenames
    }

    total_precision = 0.0
    total_recall = 0.0
    
    print("-" * 40)
    for query, relevant_docs in ground_truth.items():
        print(f"Testing Query: '{query}'")
        
        results = idx.search(query)
        retrieved_docs = []
        
        # results is a list of tuples: (docId, postings_list)
        for docId, _ in results:
            doc_name = idx.docIdMap.get(docId, "Unknown")
            retrieved_docs.append(doc_name)
            
        print(f"Retrieved {len(retrieved_docs)} docs: {retrieved_docs}")
        print(f"Relevant Expected: {relevant_docs}")
        
        # Calculate Intersection
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        true_positive = len(retrieved_set.intersection(relevant_set))
        
        precision = true_positive / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        recall = true_positive / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        total_precision += precision
        total_recall += recall
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print("-" * 40)
        
    avg_precision = total_precision / len(ground_truth)
    avg_recall = total_recall / len(ground_truth)
    
    print("\nOVERALL METRICS")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")

if __name__ == "__main__":
    evaluate()
