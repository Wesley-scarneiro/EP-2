import pandas as pd
import re
import unicodedata
import eli5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

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

# --- Modelo baseline ---
def dummy_classifier(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['req_text'], 
        df['profession'],
        test_size=0.3,
        random_state=42,
        stratify=df['profession']
    )

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    print("--- Baseline ---")
    accuracy = accuracy_score(y_test, y_pred_baseline)
    print(f"Acurácia: {accuracy:.2f}")

# --- Treinamento da Regressão Logística ---
def logistic_regression_train(df: pd.DataFrame):
    X_text = df['req_text'].values
    y_text = df['profession'].values

    # Divide os dados (70% treino / 30% teste)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_text, test_size=0.3, random_state=42, stratify=y_text
    )

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.83,
        ngram_range=(1, 4)
    )

    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        C=8.6,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAcurácia (dados de teste - 30%): {acc:.4f}")
    print("\nRelatório por classe:")
    print(classification_report(y_test, y_pred, target_names=sorted(df['profession'].unique()), digits=4))

    # Retorna tudo necessário para análise posterior
    return model, vectorizer, X_test_text, y_test, y_pred

# --- Interpretação e Análise de Erros ---
def analyze_model(model, vectorizer, X_test_text, y_test, y_pred):
    print("\n========== ANÁLISE E INTERPRETAÇÃO DO MODELO ==========\n")

    # --- (i) Interpretação Global ---
    print("--- Interpretação Global: características mais importantes por classe ---")
    weights_df = eli5.explain_weights_df(model, vec=vectorizer, top=20)
    print(weights_df.groupby('target').head(5))  # mostra top 5 termos mais relevantes por classe

    # --- (ii) Interpretação Local ---
    print("\n--- Interpretação Local (exemplo individual) ---")
    example_idx = 5  # índice de exemplo (pode mudar conforme desejar)
    explanation = eli5.explain_prediction_df(model, X_test_text[example_idx], vec=vectorizer)
    print(f"Texto de exemplo: {X_test_text[example_idx]}")
    print(f"Classe verdadeira: {y_test[example_idx]} | Classe predita: {y_pred[example_idx]}")
    print("\nPalavras com maior influência nesta predição:")
    print(explanation.head(10))

    # --- (iii) Análise de Erros ---
    print("\n--- Análise de Erros ---")
    errors_df = pd.DataFrame({
        'texto': X_test_text,
        'real': y_test,
        'predito': y_pred
    })
    erros = errors_df[errors_df['real'] != errors_df['predito']]

    print(f"Total de erros: {len(erros)} de {len(errors_df)} ({len(erros)/len(errors_df):.2%})\n")
    print("Exemplos de erros:\n")
    print(erros.sample(min(5, len(erros)), random_state=42))  # mostra até 5 exemplos aleatórios

# --- Execução principal ---
if __name__ == "__main__":
    df = read_csv('data.csv')
    dummy_classifier(df)

    model, vectorizer, X_test_text, y_test, y_pred = logistic_regression_train(df)
    analyze_model(model, vectorizer, X_test_text, y_test, y_pred)
