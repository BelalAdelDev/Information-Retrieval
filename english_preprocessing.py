from pathlib import Path
import re
import glob

def main() -> None:
    inputPath = Path(__file__).parent / "data" / "en"
    outputPath = Path(__file__).parent / "Intermediate" /"Processed Documents" / "en" 
    startPipeline(inputPath, outputPath)
    
def startPipeline(englishDocumentsPath: Path, outputPath: Path):
    documents = glob.glob(str(englishDocumentsPath) + "/**/*.txt", recursive=True)
    outputPath.mkdir(parents=True, exist_ok=True)
    
    for doc in documents:
        with open(doc, "r", encoding="utf-8") as f:
            content: str = f.read()
            contentSplit: list[str] = content.split()
            for i in range(len(contentSplit)):
                contentSplit[i] = normalize(contentSplit[i])
                contentSplit[i] = stemmer(contentSplit[i])

            preprocessedContent = removeStopWords(contentSplit)
            outputFile = outputPath / Path(doc).name
            
            with open(outputFile, "w", encoding="utf-8") as o:
                o.write(" ".join(preprocessedContent))
      
     
def removeStopWords(listOfWords: list[str]) -> list[str]:
    stopWords = r"(eg|i|me|my|myself|we|our|our|ourselves|yosu|your|yours|yourself|yourselves|he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves|what|which|who|whom|this|that|these|those|am|is|are|was|were|be|been|being|have|has|had|having|do|does|did|doing|a|an|the|and|but|if|or|because|as|until|while|of|at|by|for|with|about|against|between|into|through|during|before|after|above|below|to|from|up|down|in|out|on|off|over|under|again|further|then|once|here|there|when|where|why|how|all|any|both|each|few|more|most|other|some|such|no|nor|not|only|own|same|so|than|too|very|s|t|can|will|just|don|should|now)"
    return [word for word in listOfWords if not re.fullmatch(stopWords, word)]

def stemmer(word: str) -> str:
    word = word.lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    
    suffixRules = [
        ("ed", ""), ("ly", ""), ("es", ""), ("s", ""), ("ing", ""), ("tion", "te"),("nning", "n") ,("ization", ""), ("ational", "ate"), ("tional", "tion"), ("fulness", "ful"), ("iveness", "ive"), ("biliti", "ble"), ("ousness", "ous"), ("icate", "ic"), ("alize", "al"), ("ative", ""), ("iciti", "ic"), ("ingly", ""), ("ence", ""), ("ness", ""), ("ance", ""), ("able", ""), ("ible", ""), ("ship", ""), ("ment", ""), ("edly", "")
    ]
    for suffix,replacement in suffixRules:
        if len(word) > len(suffix) and word.endswith(suffix):
            stemmedWord = word[:-len(suffix)]+replacement
            if len(stemmedWord) >= 3:
                word = stemmedWord
                break
    return word

def normalize(word) -> str:
    clean_word = re.sub(r'(?<!\d)(?!\/\d)[^\w\s]|[^\w\s](?!\d)', '', word)
    return clean_word.lower()


if __name__ == "__main__":
    main()