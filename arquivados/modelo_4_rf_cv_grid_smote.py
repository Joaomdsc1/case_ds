import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Carregar o dataset
try:
    df = pd.read_csv('teste_lifecycledatascience.csv', sep=';')
except FileNotFoundError:
    print("Arquivo 'teste_lifecycledatascience.csv' não encontrado.")
    exit()

# 1. Limpeza e Pré-processamento
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how='all')
for col in ['Gasto_Crédito', 'Dinheiro_Investido']:
    if df[col].dtype == 'object':
        df[col] = df[col].replace({'R\$ ': '', '\.': ''}, regex=True).str.replace(',', '.').astype(float)

# 2. Engenharia de Features
limite_alto_valor = df['Gasto_Crédito'].quantile(0.75)
df['segmento'] = np.where(df['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')

# 3. Preparação para o Modelo
X = df.drop(['Cliente', 'segmento', 'Gasto_Crédito'], axis=1)
y = df['segmento']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Pipeline de Pré-processamento
categorical_features = ['Gênero']
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. CRIAÇÃO DO PIPELINE COM SMOTE
# O pipeline agora tem 3 etapas: pré-processamento, aplicação do SMOTE e o classificador.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('smote', SMOTE(random_state=42)),
                           ('classifier', RandomForestClassifier(random_state=42))])

# 6. DEFINIÇÃO DO GRID DE HIPERPARÂMETROS
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# 7. CONFIGURAÇÃO E EXECUÇÃO DO GRIDSEARCHCV COM O NOVO PIPELINE
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=2)

print("Iniciando a busca pelos melhores hiperparâmetros com SMOTE + GridSearchCV...")
grid_search.fit(X_train, y_train)

# 8. ANÁLISE DOS RESULTADOS
print("\n--- Resultados do GridSearchCV com SMOTE ---")
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# 9. AVALIAÇÃO DO MELHOR MODELO NO CONJUNTO DE TESTE
print("\n--- Avaliação Final do Melhor Modelo (com SMOTE) no Conjunto de Teste ---")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))