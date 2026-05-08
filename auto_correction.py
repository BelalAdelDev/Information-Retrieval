import numpy as np
from pathlib import Path
from positional_inverted_index import loadInvertedIndex
import json

def main() -> None:
    print(editDistance("hello", "hello"))
    k_gram_json_path: Path = Path(__file__).parent / "Intermediate" / "Inverted Index" / "KGramIndex.json"
    
    print(jaccardSimilarity([1, 2, 3, 4], [3, 4, 5, 6]))
    
# TODO: implement auto-correction logic utilizing k-gram -> edit distance -> jaccard (select highest score)
    
def buildKGramIndex(invertedIndexJsonPath) -> dict[str, list[str]]:
    invertedIndex: dict[str, dict[int, list[int]]] = loadInvertedIndex(invertedIndexJsonPath)
    kGramIndex: dict[str, list[str]] = {}
    for term in invertedIndex:
        updated_term = "$" + term + "$"
        currentGrams = [ updated_term[i: i+3] for i in range(len(updated_term) - 2) ]
        for gram in currentGrams:
            if gram not in kGramIndex:
                kGramIndex[gram] = [term]
            else:
                kGramIndex[gram].append(term)
    
    return kGramIndex
    
def loadKGramIndex(jsonPath: Path) -> dict[str, list[str]]:
    with open(jsonPath, "r", encoding="utf-8") as f:
        kGramIndex = json.load(f)
        return kGramIndex
    
def storeKGramIndex(kGramIndex, jsonPath: Path):
    jsonPath.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonPath, "w", encoding="utf-8") as f:
        json.dump(kGramIndex, f, indent=2)
    
    
def jaccardSimilarity(list1: list, list2: list):
    s1 = set(list1)
    s2 = set(list2)
    
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    
    if union == 0:
        return 0
    else:
        return intersection / union

def editDistance(word1: str, word2:str):
    N:int = len(word1) + 1
    M:int = len(word2) + 1
    
    table = np.zeros((N, M))
    for i in range(N):
        table[i][0] = i
        
    for i in range(M):
        table[0][i] = i
    
    for i in range(1, N):
        for j in range(1, M):
            if word1[i-1] == word2[j-1]:
                table[i][j] = table[i-1][j-1]
            else:
                table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
    
    return int(table[N-1][M-1])
                
if __name__ == "__main__":
    main()