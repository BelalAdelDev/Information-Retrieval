import numpy as np
from pathlib import Path
import json

def main() -> None:
    print(editDistance("hello", "hello"))
    k_gram_json_path: Path = Path(__file__).parent / "Intermediate" / "Inverted Index" / "KGramIndex.json"
    
    print(jaccardSimilarity([1, 2, 3, 4], [3, 4, 5, 6]))
    
def autoCorrect(queryTerm: str, kGramIndex: dict[str, list[str]]) -> tuple[str, int]:
    queryGrams:list[str] = getKGrams(queryTerm)
    
    foundTerms:set[str] = set()
    for queryGram in queryGrams:
        for term in kGramIndex.get(queryGram, ""):
            foundTerms.add(term) 
    
    jaccard_threshold:float = 0.1
    suggestions: dict[str, int] = {}
    
    for term in foundTerms:
        termGrams = getKGrams(term)
        if jaccardSimilarity(termGrams, queryGrams) > jaccard_threshold:
            suggestions[term] = editDistance(term, queryTerm)
    
    if suggestions != {}:
        closest_word = min(suggestions.items(), key=lambda item: item[1])
        return closest_word
    else:
        return None # type: ignore
        
def getKGrams(term: str) -> list[str]:
    updated_term = "$" + term + "$"
    return [ updated_term[i: i+3] for i in range(len(updated_term) - 2) ]

def createKGramIndex(invertedIndexJsonPath) -> dict[str, list[str]]:
    from positional_inverted_index import loadInvertedIndex
    invertedIndex: dict[str, dict[int, list[int]]] = loadInvertedIndex(invertedIndexJsonPath)
    kGramIndex: dict[str, list[str]] = {}
    for term in invertedIndex:
        currentGrams = getKGrams(term)
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


def start() -> dict[str, list[str]]:
    k_gram_json_path: Path = Path(__file__).parent / "Intermediate" / "Inverted Index" / "KGramIndex.json"
    inverted_index_json_path: Path = Path(__file__).parent / "Intermediate" / "Inverted Index" / "PositionalInvertedIndex.json"
        
    if not k_gram_json_path.exists():
        print("no saved K-Gram index, creating K-Gram index...")
        kGramIndex: dict[str, list[str]] = createKGramIndex(inverted_index_json_path)
        storeKGramIndex(kGramIndex, k_gram_json_path)
        return kGramIndex
    else:
        print("found saved K-Gram index, loading...")
        newKGramIndex: dict[str, list[str]] = loadKGramIndex(k_gram_json_path)
        return newKGramIndex
                
if __name__ == "__main__":
    main()
