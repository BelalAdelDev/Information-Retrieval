import json, glob, re
from pathlib import Path
from typing import Any
import math

class PositionalInvertedIndex:
    def __init__(self) -> None:
        self.invertedIndex, self.docIdMap = start()
        print("docIdMap: ... ", self.docIdMap)
        
    def search(self, query) -> list[tuple[int, list[int]]]: # list[str]
        # TODO: apply pipeline to query
        found: dict[int, list[int]] = self.invertedIndex[query]
        sortedFound: list[tuple[int, list[int]]] = sorted(found.items(), key=lambda item: 1 + math.log(len(item[1])), reverse=True)
        for docId, postings in sortedFound:
            for pos in postings:
                print(f"found at document {self.docIdMap[docId]} at pos {pos}")
        return sortedFound
    
    def kSearch(self, query) -> Any:
        # TODO: apply pipeline to query
        prox, normals = parseKQuery(query)

        if prox == []:
            return self.search(" ".join(normals))
        
        # TODO: K-query or K-search

def parseKQuery(query) -> Any:
    terms = query.split()
    prox: list[tuple[str, str, int]] = []
    normal: list[str] = []
    
    i: int = 0
    while i < len(terms):
        if i + 2 < len(terms) and terms[i+1].startswith("/"):
            term1 = terms[i]
            term2 = terms[i+2]
            k = int(terms[i+1][1:])
            prox.append((term1, term2, k))
            i += 3
        else:
            normal.append(terms[i])
            i += 1

    return prox, normal

    

def createInvertedIndex(docs: list[str]):
    invertedIndex: dict[str, dict[int, list[int]]] = dict()
    for docId, doc in enumerate(docs):
        words: list[str] = doc.split(" ")
        for i, word in enumerate(words):
            if word not in invertedIndex:
                invertedIndex[word] = {}
            if docId not in invertedIndex[word]:
                invertedIndex[word][docId] = []
    
            invertedIndex[word][docId].append(i)
    return invertedIndex
            


def createDocumentIdMap(documentsPath: Path) -> dict[int, str]:
    documents_paths = glob.glob(str(documentsPath) + "/**/*.txt", recursive=True)
    docIdMap: dict[int, str] = dict()
    for docId, doc_path in enumerate(documents_paths):
        docIdMap[docId] = doc_path.split("\\")[-1]
    return docIdMap    
    
def loadDocuments(documentsPath: Path) -> list[str]:
    documents_paths = glob.glob(str(documentsPath) + "/**/*.txt", recursive=True)
    documents: list[str] = []
    for doc in documents_paths:
        with open(doc, "r", encoding="utf-8") as f:
            content: str = f.read()
            documents.append(content)
    return documents
            

def storeInvertedIndex(invertedIndex, jsonPath: Path):
    jsonPath.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonPath, "w", encoding="utf-8") as f:
        json.dump(invertedIndex, f, indent=2)

def loadInvertedIndex(jsonPath: Path) -> dict[str, dict[int, list[int]]]:
    with open(jsonPath, "r", encoding="utf-8") as f:
        data = json.load(f)
        invertedIndex: dict[str, dict[int, list[int]]] = {}
        for term, postings in data.items():
            invertedIndex[term] = {}
            for docId, positions in postings.items():
                invertedIndex[term][ int(docId) ] = positions

        return invertedIndex
    
    
def start() -> Any:
    default_json_path: Path = Path(__file__).parent / "Intermediate" / "Inverted Index" / "PositionalInvertedIndex.json"
    default_doucments_path: Path = Path(__file__).parent / "Intermediate" / "Processed Documents"    
    
    docIdMap: dict[int, str] = createDocumentIdMap(default_doucments_path)
    
    if not default_json_path.exists():
        print("no saved inverted index, creating inverted index...")
        invertedIndex: dict[str, dict[int, list[int]]] = createInvertedIndex(loadDocuments(default_doucments_path))
        storeInvertedIndex(invertedIndex, default_json_path)
        return invertedIndex, docIdMap
    else:
        print("found saved inverted index, loading...")
        newInvertedIndex: dict[str, dict[int, list[int]]] = loadInvertedIndex(default_json_path)
        return newInvertedIndex, docIdMap
    
if __name__ == "__main__":
    default_json_path: Path = Path(__file__).parent / "Intermediate" / "Inverted Index" / "PositionalInvertedIndex.json"
    default_doucments_path: Path = Path(__file__).parent / "Intermediate" / "Processed Documents"
    
    ii = PositionalInvertedIndex()
    ii.search("machine")
    
    # documents = loadDocuments(default_doucments_path)
    # invertedIndex = createInvertedIndex(documents)
    
    # storeInvertedIndex(invertedIndex, default_json_path)
    # loadedInvertedIndex = loadInvertedIndex(default_json_path)
    
    # print(loadedInvertedIndex["the"] == invertedIndex["the"])
    # print(loadedInvertedIndex["تم"] == invertedIndex["تم"])

    # _, docIdMap = start()
    # print(docIdMap)