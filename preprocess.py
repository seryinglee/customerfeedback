import pandas as pd
import zipfile
import re
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if len(t) > 2
    ]

    return " ".join(tokens)


def normalize_labels(df):
    df = df.copy()

    df["review_type"] = (
        df["review_type"]
        .str.lower()
        .str.strip()
    )

    label_mapping = {
        'critcal': 'critical',
        'criticial': 'critical',
        'neg': 'negative',
        'negativ': 'negative',
        'neut': 'neutral',
        'neutal': 'neutral',
        'pos': 'positive',
        'quest': 'question',
        'q': 'question'
    }

    df["review_type"] = df["review_type"].replace(label_mapping)

    return df


def load_csv_from_zip(path="data.zip"):
    with zipfile.ZipFile(path, "r") as z:

        csv_files = [
            f for f in z.namelist()
            if f.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "No CSV file found inside data.zip"
            )

        data = pd.read_csv(z.open(csv_files[0]))

    return data


def load_data(path="data.zip"):

    data = load_csv_from_zip(path)

    df = data[
        ["description", "review_type"]
    ].dropna()

    df = normalize_labels(df)

    df["description"] = df[
        "description"
    ].apply(clean_text)

    return df


def load_raw_data(path="data.zip"):

    data = load_csv_from_zip(path)

    df = data[
        ["description", "review_type"]
    ].dropna()

    df = normalize_labels(df)

    return df
