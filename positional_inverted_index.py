import json, glob, re
from pathlib import Path
from typing import Any
import math
from arabic_preprocessing import normalize as arabic_normalizer
from arabic_preprocessing import stemmer as arabic_stemmer
from arabic_preprocessing import removeStopWords as arabic_removeStopWords
from english_preprocessing import normalize as english_normalizer
from english_preprocessing import stemmer as english_stemmer
from english_preprocessing import removeStopWords as english_removeStopWords
import auto_correction

class PositionalInvertedIndex:
    def __init__(self) -> None:
        self.invertedIndex, self.docIdMap = start()
        self.kGramIndex = auto_correction.start()
        self.docLengths = self.precomputeDocLengths()
        
    def precomputeDocLengths(self) -> dict[int, float]:
        lengths: dict[int, float] = {}
        documentCount = len(self.docIdMap)

        for _ , postings in self.invertedIndex.items():
            idf = calc_idf(len(postings), documentCount)
            
            for docId, positions in postings.items():
                weightSquared: float = calc_tf_idf(len(positions), idf) ** 2
                lengths[docId] = lengths.get(docId, 0.0) + weightSquared

        docLengths: dict[int, float] = {}
        for docId, length in lengths.items():
            docLengths[docId] = math.sqrt(length)
        return docLengths

    def buildQueryVector(self ,terms: list[str], documentCount: int) -> tuple[dict[str, float], float]:
        queryVector: dict[str, float] = {}
        queryNorm = 0.0
        for term in set(terms):
            tf = terms.count(term)
            df = len(self.invertedIndex.get(term, {}))
            idf = calc_idf(df, documentCount)
            weight = calc_tf_idf(tf, idf)
            queryVector[term] = weight
            queryNorm += weight ** 2
        
        return queryVector, math.sqrt(queryNorm)

    def search(self, query) -> list[tuple[int, list[int]]]: # list[str]
        query = applyPipeline(query)    
        queryTerms = query.split()
        
        termPostings: list[dict[int, list[int]]] = self.fetchPostingsAutoCorrect(queryTerms)
            
        commonDocumentIds: set[int] = self.commonDocumentsIds(termPostings)
        if commonDocumentIds == []:
            return []
                    
        combinedPositions: dict[int, list[int]] = self.getCombinedPositions(termPostings, commonDocumentIds)
            
        documentCount = len(self.docIdMap)
        queryVector, queryNorm = self.buildQueryVector(queryTerms, documentCount)
        
        scores = {}
        for docId in commonDocumentIds:
            dotProduct = 0.0
            for term in set(queryTerms):
                if docId in self.invertedIndex[term]:
                    tf = len(self.invertedIndex[term][docId])
                    df = len(self.invertedIndex[term])
                    idf = calc_idf(df, documentCount)
                    doc_weight = calc_tf_idf(tf, idf)
                    dotProduct += queryVector[term] * doc_weight
            
            doc_norm = self.docLengths[docId]
            if queryNorm > 0 and doc_norm > 0:
                scores[docId] = dotProduct / (queryNorm * doc_norm)
            else:
                scores[docId] = 0.0

        sortedFound = sorted(combinedPositions.items(), key=lambda item: scores[item[0]], reverse=True)
        
        for docId, postings in sortedFound:
            print(f"found at document {self.docIdMap[docId]}, Score: {scores[docId]:4f}, at pos {postings}")
        return sortedFound

    
    def fetchPostingsAutoCorrect(self, terms: list[str]):
        term_postings: list[dict[int, list[int]]] = []
        for i, term in enumerate(terms):
            postings = self.invertedIndex.get(term, {})
            if postings == {}: 
                corrected_term = auto_correction.autoCorrect(term, self.kGramIndex)
                if corrected_term != None and corrected_term[0] != term:
                    print(f"'{term}' not found. Did you mean: '{corrected_term[0]}'?")
                    terms[i] = corrected_term[0]
                    postings = self.invertedIndex[corrected_term[0]]
                    
            term_postings.append(postings)   
        return term_postings    
    
    def commonDocumentsIds(self, termPostings: list[dict[int, list[int]]]) -> set[int]:
        common_docIds:set[int] = set(termPostings[0].keys())
        for postings in termPostings[1:]:
            common_docIds = common_docIds.intersection(postings.keys())
        return common_docIds
        
    def getCombinedPositions(self, termPostings:list[dict[int, list[int]]] , commonDocumentIds: set[int]) -> dict[int, list[int]]:
        combinedPositions: dict[int, list[int]] = {}
        for docId in commonDocumentIds:
            all_pos:list[int] = []
            for postings in termPostings:
                all_pos.extend(postings[docId])
            combinedPositions[docId] = sorted(list(set(all_pos)))
        return combinedPositions
    
    def kSearch(self, query) -> Any:
        query = applyPipeline(query)
        prox, normals = parseKQuery(query)

        if prox == []:
            return self.search(" ".join(normals))
                
        prox_results: list[dict[int, list[int]]] = []
        for t1, t2, k in prox:
            p1: dict[int, list[int]] = self.invertedIndex[t1]
            p2: dict[int, list[int]] = self.invertedIndex[t2]
            
            result: dict[int, list[int]] = {}
            documentIdsIntersection = set(p1.keys()).intersection(p2.keys())
            for docId in documentIdsIntersection:
                pos1:list[int] = p1[docId]
                pos2:list[int] = p2[docId]
                matched_positions:list[int] = []
                for pp1 in pos1:
                    for pp2 in pos2:
                        if abs(pp1 - pp2) <= k:
                            matched_positions.append(pp1)
                            matched_positions.append(pp2)
                        
                if matched_positions != []:
                    result[docId] = sorted(list(set(matched_positions)))
            prox_results.append(result)
            
        normal_results: list[dict[int, list[int]]] = []
        for term in normals:
            normal_results.append(self.invertedIndex[term])
            
        all_results: list[dict[int, list[int]]] = prox_results + normal_results
        
        if all_results == []:
            return []
            
        common_docIds = set(all_results[0].keys())
        for res in all_results[1:]:
            common_docIds = common_docIds.intersection(res.keys())
            
        final_found: dict[int, list[int]] = {}
        for docId in common_docIds:
            all_pos: list[int] = []
            for res in all_results:
                all_pos.extend(res[docId])
            final_found[docId] = sorted(list(set(all_pos)))
            
        sortedFound: list[tuple[int, list[int]]] = sorted(final_found.items(), key=lambda item: 1 + math.log(len(item[1])), reverse=True)
        for docId, postings in sortedFound:
            for pos in postings:
                print(f"found at document {self.docIdMap[docId]} at pos {pos}")
                
        return sortedFound



def calc_idf(df: int, documentCount: int) -> float:
    if df > 0:
        return math.log10(documentCount / float(df))
    else:
        return 0.0
    

def calc_tf_idf(tf: int, idf: float) -> float:
    if tf > 0:
        return (1 + math.log10(tf)) * idf
    else:
        return 0.0

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

def applyPipeline(query: str) -> str:
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    words:list[str] = query.split()
    processed_words: list[str] = []
    
    for word in words:
        if is_arabic:
            w = arabic_normalizer(word)
            w = arabic_stemmer(w)
        else:
            w = english_normalizer(word)
            w = english_stemmer(w)
        if w:
            processed_words.append(w)
        
    if is_arabic:
        processed_words = arabic_removeStopWords(processed_words)
    else:
        processed_words = english_removeStopWords(processed_words)
        
    return " ".join(processed_words)

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