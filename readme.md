# Trabalho de PLN - Processamento de Linguagem Natural - EP2

Os seguintes passos devem ser realizados para executar o EP corretamente:
1. Criar um ambiente virtual python (python -m venv venv)
2. Instalar as dependências via requirements.txt (pip install -r requirements.txt)
3. Executar o arquivo run_gridSearch.py.py’ para pesquisa dos melhores parâmetros
4. Executar o arquivo train_model.py.py’ com os parâmetros ótimos encontratos

## Melhor combinação de parâmetros:
```json
{
  "clf__C": 8.60,
  "clf__class_weight": "balanced",
  "tfidf__max_df": 0.83,
  "tfidf__min_df": 2,
  "tfidf__ngram_range": [1, 4],
  "tfidf__sublinear_tf": true
}
```

* Melhor acurácia obtida (validação cruzada): 0.6998
* Acurácia no conjunto de teste: 0.7015

## Relatório de classificação

```
              precision    recall  f1-score   support

    academic       0.71      0.69      0.70      4378
  government       0.76      0.74      0.75      5635
     private       0.60      0.64      0.62      3091

    accuracy                           0.70     13104
   macro avg       0.69      0.69      0.69     13104
weighted avg       0.70      0.70      0.70     13104
```

## 🧩 Desempenho do Modelo

**Baseline:** 0.43  
**Acurácia (dados de teste - 30%)**: 0.7037

---

## 📊 Relatório por Classe

```
              precision    recall  f1-score   support

    academic     0.7116    0.6976    0.7045      4378
  government     0.7620    0.7398    0.7508      5635
     private     0.5980    0.6464    0.6213      3091

    accuracy                         0.7037     13104
   macro avg     0.6905    0.6946    0.6922     13104
weighted avg     0.7065    0.7037    0.7048     13104
```

---

## 🔍 Interpretação e Análise do Modelo

### Interpretação Global — Características mais importantes por classe
```
        target            feature    weight
0     academic          professor  5.779542
1     academic           pesquisa  4.737362
2     academic           obrigada  4.507279
3     academic              dados  4.280683
4     academic          estudante  4.050540
20  government     redistribuicao  6.767933
21  government           servidor  5.420249
22  government         servidores  5.414577
23  government           solicito  4.615862
24  government  solicito informar  3.720078
40     private            empresa  4.372895
41     private           obrigado  3.371279
42     private            produto  3.249317
43     private               onde  3.202226
44     private             de uso  3.142881
```

---

### Interpretação Local — Exemplo individual

**Texto de exemplo:**  
> trata este pedido de informacoes sobre o projeto territorios do axe anexo de estudo sobre as religioes de matriz africana bem como mapeamento das casas que as praticam por toda a regiao da grande florianopolis conforme site tendo em vista que ainda nao estao no portal da fapeu os relatorios parciais de execucao financeira e execucao tecnica do primeiro semestre de conforme gostaria que me fossem enviadas copias digitalizadas da documentacao relativa a o processo que contem o projeto original da ufsc chamado territorios do axe o processo de contratacao da fapeu pela ufsc para execucao desse projeto inclusive copias digitalizadas dos documentos de execucao notas fiscais faturas extratos etc que ainda estejam na fapeu e por algum motivo ainda nao tenham sido enviados a ufsc para publicacao em tempo real como determina a legislacao de transparencia aguardo resposta att

**Classe verdadeira:** `government`  
**Classe predita:** `government`

**Palavras com maior influência nesta predição:**
```
     target        feature    weight     value
0  academic        projeto  0.187003  0.085399
1  academic    territorios  0.136710  0.125586
2  academic      regiao da  0.112323  0.074173
3  academic       execucao  0.087324  0.115894
4  academic     relatorios  0.075133  0.050690
5  academic     mapeamento  0.074188  0.065599
6  academic  legislacao de  0.072904  0.077967
7  academic            att  0.069670  0.037807
8  academic       estao no  0.068787  0.072211
9  academic          ainda  0.061153  0.077880
```

---

## ❌ Análise de Erros

**Total de erros:** 3883 de 13104 (29.63%)

**Exemplos de erros:**
```
                                                   texto      real     predito
4581   solicito relacao nominal de todos os servidore...   private  government
8936   prezados diante do decreto que no seu art o fi...   private  government
12230  ola boa tarde gostaria de receber em minha res...  academic  government
12264  trabalho no departamento fiscal de uma empresa...  academic     private
6146   prezados gostaria de saber quais os motivos qu...  academic  government
```
