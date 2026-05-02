from pathlib import Path
import re
import glob

def main() -> None:
    inputPath = Path(__file__).parent / "data" / "ar"
    outputPath = Path(__file__).parent / "Intermediate" /"Processed Documents" / "ar" 
    startPipeline(inputPath, outputPath)
    
def startPipeline(arabicDocumentsPath: Path, outputPath: Path):
    documents = glob.glob(str(arabicDocumentsPath) + "/**/*.txt", recursive=True)
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
    stopWords = r"(الذين|لماذا|انتما|اليوم|هؤلاء|عندما|والذي|والتي|وكانت|انتم|انتن|كانت|يكون|الان|ايضا|الذي|ماذا|التي|امام|كلما|هناك|وكان|اذا|بعض|كان|كما|لكن|غدا|حول|كيف|ضمن|لهم|تلك|غير|ذلك|هنا|جدا|عند|حيث|دون|اين|بين|فقط|اذن|ليس|لها|حين|وهي|بعد|هذا|وقد|امس|انا|قبل|هذه|نحن|هما|انت|بها|اما|وهو|هي|كل|به|اي|مع|هل|او|هم|له|في|هن|ما|لا|بل|لي|قد|لن|عن|اذ|من|ان|هو|لم|ثم|ل|و|ك|ب)"
    return [word for word in listOfWords if not re.fullmatch(stopWords, word)]

def stemmer(word: str) -> str:
    PREFIXES = ["وال", "بال", "كال", "فال", "لل","و", "ف", "ب", "ك", "ل", "ال"]
    SUFFIXES = ["يات", "ات", "ون", "ين", "ان","ه", "ها", "ك", "ي", "كم", "نا","ة", "ت", "ا"]
    
    for prefix in PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix):]
            break
    
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
            
    return word

def normalize(word) -> str:
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي"
    }
    for origin, replacement in replacements.items():
        word = word.replace(origin, replacement)
        
    word = re.sub(r'[\u064B-\u0652\u06D6-\u06ED]', '', word)
    clean_word = re.sub(r'(?<!\d)[^\w\s]|[^\w\s](?!\d)', '', word)
    return clean_word.lower()


if __name__ == "__main__":
    main()