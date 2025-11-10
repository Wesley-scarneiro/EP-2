import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint

# --- Função de limpeza de texto ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'http\S+|www\S+', ' ', text)  # remove URLs
    text = re.sub(r'\d+', ' ', text)  # remove números
    text = re.sub(r"[;:,.!?()\[\]\"'—–\-_/\\]", " ", text)  # remove pontuação
    text = re.sub(r'\s+', ' ', text).strip()  # remove espaços extras
    return text

# --- Carrega CSV e limpa texto ---
def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df.dropna(subset=['req_text'], inplace=True)
    df['req_text'] = df['req_text'].apply(clean_text)
    return df


# --- Regressão Logística com RandomizedSearchCV ---
def logistic_regression_tfidf_randomsearch(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['req_text'],
        df['profession'],
        test_size=0.3,
        random_state=42,
        stratify=df['profession']
    )

    # Pipeline: TF-IDF + Regressão Logística
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42, max_iter=500))
    ])

    # Espaço de busca de parâmetros (amostrado aleatoriamente)
    param_distributions = {
        'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1, 4)],
        'tfidf__min_df': randint(2, 6),
        'tfidf__max_df': uniform(0.7, 0.3),  # valores entre 0.7 e 1.0
        'tfidf__sublinear_tf': [True, False],
        'clf__C': uniform(0.1, 10.0),  # C entre 0.1 e 10
        'clf__class_weight': ['balanced']
    }

    # Validação cruzada estratificada (10 folds)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Busca aleatória de hiperparâmetros
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=30,              # número de combinações testadas
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    random_search.fit(X_train, y_train)

    # Resultados
    print("\n✅ Melhor combinação de parâmetros:")
    print(random_search.best_params_)

    print(f"\n🔹 Melhor acurácia obtida (validação cruzada): {random_search.best_score_:.4f}")

    # Avaliação final no conjunto de teste
    y_pred = random_search.best_estimator_.predict(X_test)
    print(f"\n🎯 Acurácia no conjunto de teste: {accuracy_score(y_test, y_pred):.4f}")
    print("\n📊 Relatório de classificação:")
    print(classification_report(y_test, y_pred))

    return random_search.best_estimator_


# --- Execução ---
if __name__ == "__main__":
    df = read_csv('data.csv')
    best_model = logistic_regression_tfidf_randomsearch(df)
