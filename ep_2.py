#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# ===== NOVO: gensim para Word2Vec =====
from gensim.models import Word2Vec


# -------------------------
# Padrões de limpeza
# -------------------------

URL_PATTERN = re.compile(
    r"""(?xi)
    \b
    (?:https?://|www\.)
    [^\s<>"]+        # tudo até espaço/aspas/<>
    """
)

PUNCT_PATTERN = re.compile(r"[;:,.!?()\[\]\"'—–\-]")


# -------------------------
# IO de dados
# -------------------------

def read_file() -> Optional[pd.DataFrame]:
    """
    Lê 'data.csv' (no mesmo diretório do script) usando cp1252 e ';'.
    Espera colunas: req_text, profession
    """
    data_file = Path(__file__).with_name('data.csv')

    try:
        df = pd.read_csv(data_file, encoding='cp1252', sep=';')
    except FileNotFoundError:
        print(f"[ERRO] data file not found: {data_file}", file=sys.stderr)
        return None
    except UnicodeDecodeError as e:
        print(f"[ERRO] unicode decode error reading data.csv with cp1252: {e}", file=sys.stderr)
        return None
    except pd.errors.EmptyDataError:
        print(f"[ERRO] data.csv is empty: {data_file}", file=sys.stderr)
        return None
    except pd.errors.ParserError as e:
        print(f"[ERRO] error parsing data.csv (sep=';'): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[ERRO] unexpected error reading data.csv: {e}", file=sys.stderr)
        return None

    return df


# -------------------------
# Limpeza básica do texto
# -------------------------

def clean_text(text: str) -> str:
    """
    Limpeza leve: remove URLs e normaliza espaços.
    Mantém acentos (útil para PLN em PT).
    """
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub(" ", text)              # remove URLs (http/https/www)
    text = re.sub(r"\s+", " ", text).strip()       # normaliza espaços
    return text


# Tokenização simples (palavras e dígitos)
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def preparing_data(clean: bool = False) -> Optional[pd.DataFrame]:
    df = read_file()
    if df is None:
        return None

    df = df.dropna(subset=['req_text']).copy()
    df['req_text'] = df['req_text'].astype(str).str.lower()

    if clean:
        df['req_text'] = df['req_text'].apply(clean_text)

    df = df.dropna(subset=['profession'])

    print("[INFO] Amostra dos dados:")
    print(df.head(3))
    print("\n[INFO] Distribuição de classes (%):")
    print((df['profession'].value_counts(normalize=True) * 100).round(2))

    return df


# -------------------------
# Baseline TF-IDF + LogReg (o seu pipeline)
# -------------------------

def train_evaluate_kfold(df: pd.DataFrame, k: int = 10) -> None:
    """
    Executa cross-validation estratificada (k-fold) com TF-IDF + Regressão Logística.
    Imprime média e desvio-padrão para accuracy e F1-macro.
    """
    X = df["req_text"].values
    y = df["profession"].values

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            solver='lbfgs',
            multi_class='auto'
        ))
    ])

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    print(f"\n[INFO] Rodando StratifiedKFold (k={k}) - TFIDF+LogReg...\n")
    res = cross_validate(pipe, X, y, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1)

    acc_scores = res['test_accuracy']
    f1_scores = res['test_f1_macro']

    print("================= RESULTADOS K-FOLD (TF-IDF) =================")
    print(f"Accuracy (mean ± std) .........: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"F1-score (macro) (mean ± std) .: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


# -------------------------
# NOVO: Word2Vec + LogReg
# -------------------------

class W2VVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformer scikit-learn que treina Word2Vec no fit() (apenas nos dados de treino do fold)
    e transforma documentos em embeddings pela média dos vetores de palavras.
    Opcionalmente, pondera por IDF (TF-IDF weighting).

    Parâmetros:
      - vector_size: dimensionalidade dos vetores
      - window, min_count, sg, epochs: hiperparâmetros do Word2Vec
      - workers: threads internas do gensim (mantenha 1 quando usar cross_validate(n_jobs>1))
      - seed: reprodutibilidade
      - use_tfidf_weighting: se True, pondera as palavras por IDF
    """
    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,           # 1=skip-gram, 0=CBOW
        epochs: int = 10,
        workers: int = 1,
        seed: int = 42,
        use_tfidf_weighting: bool = False
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.workers = workers
        self.seed = seed
        self.use_tfidf_weighting = use_tfidf_weighting
        self.model_ = None
        self.kv_ = None
        self.idf_: Dict[str, float] = {}

    def _prepare_corpus(self, X) -> List[List[str]]:
        corpus = []
        for doc in X:
            tokens = simple_tokenize(clean_text(doc))
            corpus.append(tokens)
        return corpus

    def fit(self, X, y=None):
        corpus = self._prepare_corpus(X)

        # Treina Word2Vec apenas com os dados do fold de treino
        self.model_ = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        self.kv_ = self.model_.wv

        # Se for usar peso por IDF, calculamos aqui (usando TfidfVectorizer para obter idf_)
        if self.use_tfidf_weighting:
            # O TfidfVectorizer espera strings; juntamos tokens com espaço
            docs_as_text = [" ".join(toks) for toks in corpus]
            tfidf = TfidfVectorizer(min_df=1)
            tfidf.fit(docs_as_text)
            # Mapeia termo -> idf
            vocab = tfidf.vocabulary_
            idf_vals = tfidf.idf_
            self.idf_ = {term: idf_vals[idx] for term, idx in vocab.items()}
        else:
            self.idf_ = {}

        return self
    def _doc_vector(self, tokens: List[str]) -> np.ndarray:
        # Se nenhum termo com vetor, retorna zero-vector
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)

        if not self.use_tfidf_weighting:
            # média simples
            vecs = [self.kv_[t] for t in tokens if t in self.kv_]
            if not vecs:
                return np.zeros(self.vector_size, dtype=np.float32)
            return np.mean(vecs, axis=0)

        # média ponderada por IDF
        weighted_sum = np.zeros(self.vector_size, dtype=np.float32)
        weight_total = 0.0
        for t in tokens:
            if t in self.kv_:
                w = self.idf_.get(t, 1.0)
                weighted_sum += self.kv_[t] * w
                weight_total += w
        if weight_total == 0.0:
            return np.zeros(self.vector_size, dtype=np.float32)
        return weighted_sum / weight_total

    def transform(self, X):
        vectors = []
        for doc in X:
            tokens = simple_tokenize(clean_text(doc))
            vectors.append(self._doc_vector(tokens))
        return np.vstack(vectors)


def train_evaluate_kfold_w2v(
    df: pd.DataFrame,
    k: int = 10,
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    epochs: int = 10,
    workers: int = 1,
    seed: int = 42,
    use_tfidf_weighting: bool = False,
) -> None:
    """
    Executa cross-validation estratificada (k-fold) com pipeline:
        [W2VVectorizer -> LogisticRegression]
    Imprime média e desvio-padrão para accuracy e F1-macro.
    """
    X = df["req_text"].values
    y = df["profession"].values
    emb = W2VVectorizer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers,  # mantenha 1 se parallelizar os folds (n_jobs=-1)
        seed=seed,
        use_tfidf_weighting=use_tfidf_weighting,
    )

    pipe = Pipeline(steps=[
        ("w2v", emb),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            random_state=seed
        ))
    ])

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    print(f"\n[INFO] Rodando StratifiedKFold (k={k}) - W2V+LogReg (idf_weight={use_tfidf_weighting})...\n")
    res = cross_validate(
        pipe, X, y,
        scoring=scoring,
        cv=cv,
        return_train_score=False,
        n_jobs=-1  # paraleliza por fold; cuidado para não somar com workers>1 no gensim
    )

    acc_scores = res['test_accuracy']
    f1_scores = res['test_f1_macro']

    print("================= RESULTADOS K-FOLD (WORD2VEC) =================")
    print(f"Accuracy (mean ± std) .........: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"F1-score (macro) (mean ± std) .: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")


# -------------------------
# MAIN
# -------------------------

def main() -> int:
    # Se quiser aplicar a limpeza básica ANTES de tudo, troque para True
    clean = False

    df = preparing_data(clean)
    if df is None:
        return 1

    # 1) Baseline TF-IDF + LogReg
    train_evaluate_kfold(df, k=10)

    # 2) Word2Vec + LogReg (média simples)
    train_evaluate_kfold_w2v(
        df,
        k=10,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,            # skip-gram (costuma ir melhor semanticamente)
        epochs=10,
        workers=1,       # mantenha 1 pois os folds já rodam em paralelo (n_jobs=-1)
        seed=42,
        use_tfidf_weighting=False
    )

    # 3) (Opcional) Word2Vec + LogReg ponderado por IDF
    train_evaluate_kfold_w2v(
        df,
        k=10,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,
        epochs=10,
        workers=1,
        seed=42,
        use_tfidf_weighting=True
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())