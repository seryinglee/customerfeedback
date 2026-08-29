from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import preprocess
import re

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


class BERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=None, epochs=3, batch_size=16, lr=2e-5, max_len=128):
        super(BERTClassifier, self).__init__()

        if num_labels is None:
            num_labels = 2  

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        out = self.dropout(cls_output)
        return self.linear(out)

    def fit(self, X_train, y_train):
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train) 
        
        print(f"Training with {len(self.label_encoder.classes_)} classes: {list(self.label_encoder.classes_)}")

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        encodings = self.tokenizer(
            list(X_train),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(y_train_encoded) 
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.train()
        for epoch in range(self.epochs):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in loop:
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
                
    def predict_with_confidence(self, X, confidence_threshold=0.7):
        encodings = self.tokenizer(
            list(X),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        all_probs = []
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [x.to(self.device) for x in batch]
                outputs = self(input_ids, attention_mask)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)

        predictions = []
        for prob in all_probs:
            max_prob = np.max(prob)
            if max_prob < confidence_threshold:
                predictions.append("neutral")
            else:
                pred_class = np.argmax(prob)
                predictions.append(self.label_encoder.inverse_transform([pred_class])[0])
    
        return predictions

    def predict_with_rules(self, X, confidence_threshold=0.7):
        predictions = self.predict_with_confidence(X, confidence_threshold)
        
        neutral_patterns = [
            r"nothing special",
            r"not bad.*not great",
            r"it's okay$", 
            r"^works as expected",
            r"average product",
            r"does the job"
        ]
        
        negative_phrases = [
            "not ok", "not okay", "not good", "not great", 
            "bad", "terrible", "awful", "horrible", "disappointing"
        ]
        
        final_predictions = []
        for text, bert_pred in zip(X, predictions):
            text_lower = text.lower()
            
            is_neutral = any(re.search(pattern, text_lower) for pattern in neutral_patterns)
            is_negative = any(neg_word in text_lower for neg_word in negative_phrases)
            
            if is_neutral and not is_negative:
                final_predictions.append("neutral")
            else:
                final_predictions.append(bert_pred)
        
        return final_predictions

    def predict(self, X):
        return self.predict_with_rules(X)


def load_dataset(for_bert=False):
    if for_bert:
        df = preprocess.load_raw_data()
    else:
        df = preprocess.load_data()

    if df.empty:
        print("❌ No data available for training.")
        return None, None
    
    samples_per_class = 500
    balanced_df = df.groupby(
        "review_type",
        group_keys=False
    ).apply(
        lambda x: x.sample(
            n=min(len(x), samples_per_class),
            random_state=42
        )
    ).reset_index(drop=True)
    
    return balanced_df["description"], balanced_df["review_type"]

def train_logreg_model():
    X, y = load_dataset(for_bert=False)
    if X is None: return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)

    return model, (X_test, y_test)

@st.cache_resource
def train_svm_model():
    print("Training SVM model...")

    X, y = load_dataset(for_bert=False)
    if X is None:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
        lowercase=True,
        token_pattern=r"\b[a-zA-Z]{2,}\b"
    )

    svm_clf = SVC(
        kernel="linear",
        C=0.8,
        probability=False,
        class_weight="balanced",
        cache_size=1000,
        max_iter=1500,
        random_state=42,
        tol=1e-4
    )

    model = Pipeline([
        ("tfidf", tfidf),
        ("clf", svm_clf)
    ])

    print("Fitting SVM model...")
    model.fit(X_train, y_train)

    print("SVM training completed!")

    return model, (X_test, y_test)

@st.cache_resource
def train_bert_model(for_bert=True):
    X, y = load_dataset(for_bert=True)

    if X is None:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    num_classes = y.nunique()

    model = BERTClassifier(
        epochs=3,
        num_labels=num_classes
    )

    model.fit(X_train, y_train)

    return model, (X_test, y_test)
    
def evaluate_models():
    X_raw, y_raw = load_dataset(for_bert=True)
    if X_raw is None:
        return pd.DataFrame()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    X_train_cleaned = X_train_raw.apply(preprocess.clean_text) 
    X_test_cleaned = X_test_raw.apply(preprocess.clean_text)

    results = []

    logreg_model, _ = train_logreg_model()
    svm_model, _ = train_svm_model()
    bert_model, _ = train_bert_model()

    models = {
        "Logistic Regression": logreg_model,
        "SVM": svm_model,
        "BERT": bert_model
    }

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        if name == "BERT":
            y_pred = model.predict(X_test_raw)   
        else:
            y_pred = model.predict(X_test_cleaned)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test_raw, y_pred),
            "Precision": precision_score(y_test_raw, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test_raw, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test_raw, y_pred, average="weighted", zero_division=0),
        })

    return pd.DataFrame(results)

