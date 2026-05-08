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
    engine = SearchEngine()
    run_app(engine)

if __name__ == "__main__":
    main()
