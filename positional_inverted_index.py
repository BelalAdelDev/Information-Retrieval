import json, glob, re
from pathlib import Path
from typing import Any
import math
import english_preprocessing
import arabic_preprocessing
import auto_correction


def apply_pipeline(query: str) -> str:
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))
    words = query.split()
    processed_words = []
    
    for word in words:
        if is_arabic:
            w = arabic_preprocessing.normalize(word)
            w = arabic_preprocessing.stemmer(w)
        else:
            w = english_preprocessing.normalize(word)
            w = english_preprocessing.stemmer(w)
        if w:
            processed_words.append(w)
        
    if is_arabic:
        processed_words = arabic_preprocessing.removeStopWords(processed_words)
    else:
        processed_words = english_preprocessing.removeStopWords(processed_words)
        
    return " ".join(processed_words)

class PositionalInvertedIndex:
    def __init__(self) -> None:
        self.invertedIndex, self.docIdMap = start()
        print("docIdMap: ... ", self.docIdMap)
        
        # Precompute document vector lengths for Cosine Similarity
        self.docLengths = {} #type: ignore
        total_docs = len(self.docIdMap)
        if total_docs > 0:
            for term, postings in self.invertedIndex.items():
                df = len(postings)
                idf = math.log10(total_docs / float(df)) if df > 0 else 0.0
                for docId, positions in postings.items():
                    tf = len(positions)
                    weight = (1 + math.log10(tf)) * idf
                    self.docLengths[docId] = self.docLengths.get(docId, 0.0) + weight ** 2
            
            for docId in self.docLengths:
                self.docLengths[docId] = math.sqrt(self.docLengths[docId])
        
        # Build k-gram index for spelling correction
        self.kGramIndex = {} #type: ignore
        for term in self.invertedIndex:
            updated_term = "$" + term + "$"
            currentGrams = [updated_term[i: i+3] for i in range(len(updated_term) - 2)]
            for gram in currentGrams:
                if gram not in self.kGramIndex:
                    self.kGramIndex[gram] = [term]
                else:
                    self.kGramIndex[gram].append(term)
        
    def search(self, query) -> list[tuple[int, list[int]]]:
        # TODO: apply pipeline to query
        processed_query = apply_pipeline(query)
        if not processed_query:
            return []
            
        terms = processed_query.split()
        
        # 1. Fetch postings for all terms
        term_postings = []
        for i, term in enumerate(terms):
            postings = self.invertedIndex.get(term, {})
            if not postings: 
                # Attempt spelling correction
                corrected_term = auto_correction.autoCorrect(term, self.kGramIndex)
                if corrected_term and corrected_term != term:
                    print(f"'{term}' not found. Did you mean: '{corrected_term}'?")
                    terms[i] = corrected_term
                    postings = self.invertedIndex.get(corrected_term, {})
                    
            if not postings:
                return [] # Intersection is empty if any term is missing even after autocorrect
            term_postings.append(postings)
            
        # 2. Intersect document IDs
        common_docIds = set(term_postings[0].keys())
        for postings in term_postings[1:]:
            common_docIds = common_docIds.intersection(postings.keys())
            
        if not common_docIds:
            return []
            
        # 3. Format the output 
        final_found: dict[int, list[int]] = {}
        for docId in common_docIds:
            all_pos = []
            for postings in term_postings:
                all_pos.extend(postings[docId])
            final_found[docId] = sorted(list(set(all_pos)))
            
        # 4. Calculate TF-IDF & Cosine Similarity
        total_docs = len(self.docIdMap)
        scores = {}
        
        # Calculate Query Vector
        query_vector = {}
        query_norm = 0.0
        for term in set(terms):
            tf = terms.count(term)
            df = len(self.invertedIndex.get(term, {}))
            idf = math.log10(total_docs / float(df)) if df > 0 else 0.0
            weight = (1 + math.log10(tf)) * idf
            query_vector[term] = weight
            query_norm += weight ** 2
        query_norm = math.sqrt(query_norm)
        
        # Calculate Cosine Similarity
        for docId in common_docIds:
            dot_product = 0.0
            for term in set(terms):
                if docId in self.invertedIndex.get(term, {}):
                    tf = len(self.invertedIndex[term][docId])
                    df = len(self.invertedIndex[term])
                    idf = math.log10(total_docs / float(df)) if df > 0 else 0.0
                    doc_weight = (1 + math.log10(tf)) * idf
                    dot_product += query_vector[term] * doc_weight
            
            doc_norm = self.docLengths.get(docId, 1.0)
            if query_norm > 0 and doc_norm > 0:
                scores[docId] = dot_product / (query_norm * doc_norm)
            else:
                scores[docId] = 0.0

        # Sort based on the computed cosine similarity score
        sortedFound = sorted(final_found.items(), key=lambda item: scores.get(item[0], 0.0), reverse=True)
        
        for docId, postings in sortedFound:
            print(f"found at document {self.docIdMap[docId]} (Score: {scores.get(docId, 0.0):.4f}) at pos {postings}")
        return sortedFound
    
    def kSearch(self, query) -> Any:
        # TODO: apply pipeline to query
        prox, normals = parseKQuery(query)
        
        normals_str = apply_pipeline(" ".join(normals))
        normals = normals_str.split() if normals_str else []
        
        processed_prox = []
        for t1, t2, k in prox:
            pt1 = apply_pipeline(t1)
            pt2 = apply_pipeline(t2)
            processed_prox.append((pt1, pt2, k))
        prox = processed_prox

        if prox == []:
            return self.search(" ".join(normals))
        
        # TODO: K-query or K-search
        prox_results: list[dict[int, list[int]]] = []
        for t1, t2, k in prox:
            p1 = self.invertedIndex.get(t1, {})
            p2 = self.invertedIndex.get(t2, {})
            
            result = {}
            for docId in set(p1.keys()).intersection(p2.keys()):
                pos1 = p1[docId]
                pos2 = p2[docId]
                matched_positions = []
                for pp1 in pos1:
                    for pp2 in pos2:
                        if abs(pp1 - pp2) <= k:
                            matched_positions.append(pp1)
                            matched_positions.append(pp2)
                if matched_positions:
                    result[docId] = sorted(list(set(matched_positions)))
            prox_results.append(result)
            
        normal_results: list[dict[int, list[int]]] = []
        for term in normals:
            normal_results.append(self.invertedIndex.get(term, {}))
            
        all_results = prox_results + normal_results
        if not all_results:
            return []
            
        common_docIds = set(all_results[0].keys())
        for res in all_results[1:]:
            common_docIds = common_docIds.intersection(res.keys())
            
        final_found: dict[int, list[int]] = {}
        for docId in common_docIds:
            all_pos = []
            for res in all_results:
                all_pos.extend(res[docId])
            final_found[docId] = sorted(list(set(all_pos)))
            
        sortedFound: list[tuple[int, list[int]]] = sorted(final_found.items(), key=lambda item: 1 + math.log(len(item[1])), reverse=True)
        for docId, postings in sortedFound:
            for pos in postings:
                print(f"found at document {self.docIdMap[docId]} at pos {pos}")
                
        return sortedFound

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