import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Função para criar as novas features
def feature_engineering(df):
    df_copy = df.copy()
    df_copy['Idade'] = df_copy['Idade'].replace(0, 1) # Evitar divisão por zero
    df_copy['investimento_por_idade'] = df_copy['Dinheiro_Investido'] / df_copy['Idade']
    df_copy['interacao_investimento_idade'] = df_copy['Dinheiro_Investido'] * df_copy['Idade']
    return df_copy

# Carregar o dataset
try:
    df = pd.read_csv('teste_lifecycledatascience.csv', sep=';')
except FileNotFoundError:
    print("Arquivo 'teste_lifecycledatascience.csv' não encontrado.")
    exit()

# Limpeza e pré-processamento inicial
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how='all')
for col in ['Gasto_Crédito', 'Dinheiro_Investido']:
    if df[col].dtype == 'object':
        df[col] = df[col].replace({'R\$ ': '', '\.': ''}, regex=True).str.replace(',', '.').astype(float)

# Criar a variável alvo
limite_alto_valor = df['Gasto_Crédito'].quantile(0.75)
df['segmento'] = np.where(df['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')

# 1. APLICAR A ENGENHARIA DE FEATURES DIRETAMENTE
X = df.drop(['Cliente', 'segmento', 'Gasto_Crédito'], axis=1)
y = df['segmento']
X_eng = feature_engineering(X) # Aplicando a criação de features

# Dividir os dados já com as novas features
X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42, stratify=y)


# 2. CRIAR PIPELINE SIMPLIFICADO
# Identificar colunas para o pré-processamento
categorical_features = ['Gênero']
numeric_features = X_eng.select_dtypes(include=np.number).columns.tolist()

# Pré-processador que lida com todas as features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Pipeline com 3 etapas: Pré-processamento, SMOTE e Classificador
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])


# 3. GRIDSEARCHCV NO PIPELINE SIMPLIFICADO
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__num_leaves': [20, 31, 40],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=2)

print("Iniciando a busca com Features de Engenharia (sem seleção)...")
grid_search.fit(X_train, y_train)

# 4. ANÁLISE DOS RESULTADOS
print("\n--- Resultados do GridSearchCV (Eng. Features, sem Seleção) ---")
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# Avaliação do melhor modelo
print("\n--- Avaliação Final do Melhor Modelo ---")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))