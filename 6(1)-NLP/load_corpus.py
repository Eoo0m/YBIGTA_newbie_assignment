from datasets import load_dataset


def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    corpus = [item["verse_text"] for item in dataset if item["verse_text"]][:50]

    return corpus
