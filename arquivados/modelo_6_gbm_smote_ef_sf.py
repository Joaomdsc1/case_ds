import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 1. ETAPA DE ENGENHARIA DE FEATURES
# Criamos uma função para adicionar as novas colunas.
def feature_engineering(df):
    df_copy = df.copy()
    df_copy['Idade'] = df_copy['Idade'].replace(0, 1)
    
    # Criando novas features
    df_copy['investimento_por_idade'] = df_copy['Dinheiro_Investido'] / df_copy['Idade']
    df_copy['interacao_investimento_idade'] = df_copy['Dinheiro_Investido'] * df_copy['Idade']
    return df_copy

# Carregar o dataset
try:
    df = pd.read_csv('teste_lifecycledatascience.csv', sep=';')
except FileNotFoundError:
    print("Arquivo 'teste_lifecycledatascience.csv' não encontrado.")
    exit()

# Limpeza básica
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how='all')
for col in ['Gasto_Crédito', 'Dinheiro_Investido']:
    if df[col].dtype == 'object':
        df[col] = df[col].replace({'R\$ ': '', '\.': ''}, regex=True).str.replace(',', '.').astype(float)

# Criar a variável alvo
limite_alto_valor = df['Gasto_Crédito'].quantile(0.75)
df['segmento'] = np.where(df['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')

# Preparação para o Modelo
X = df.drop(['Cliente', 'segmento', 'Gasto_Crédito'], axis=1)
y = df['segmento']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 2. CRIAÇÃO DO PIPELINE COMPLETO
# Identificando as colunas originais
categorical_features = ['Gênero']
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# Transformador para engenharia de features
feature_creator = FunctionTransformer(feature_engineering)

# Pré-processador para as colunas originais
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough') # 'passthrough' mantém as novas colunas criadas


# Pipeline principal com 5 etapas
pipeline = Pipeline(steps=[
    ('feature_creator', feature_creator),
    ('preprocessor', preprocessor), # Etapa de pré-processamento é omitida aqui pois será feita dentro de um novo ColumnTransformer
    ('smote', SMOTE(random_state=42)),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

# 3. GRIDSEARCHCV
# O grid pode ser expandido para testar parâmetros das outras etapas também
param_grid = {
    'feature_selection__threshold': ['mean', 'median'], # Testar diferentes limiares para seleção
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__num_leaves': [20, 31],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=2)

print("Iniciando a busca com Engenharia e Seleção de Features...")
# O pipeline irá primeiro criar as features e depois aplicar as transformações
# A forma mais fácil é aplicar a engenharia de features antes e recriar o preprocessor
X_train_eng = feature_engineering(X_train)
X_test_eng = feature_engineering(X_test)

new_numeric_features = X_train_eng.select_dtypes(include=np.number).columns.tolist()

# Recriar preprocessor para incluir as novas features
preprocessor_eng = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), new_numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

# Recriar pipeline sem o feature_creator, pois já foi aplicado
pipeline_final = Pipeline(steps=[
    ('preprocessor', preprocessor_eng),
    ('smote', SMOTE(random_state=42)),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

grid_search_final = GridSearchCV(pipeline_final, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=2)

grid_search_final.fit(X_train_eng, y_train)

# 4. ANÁLISE DOS RESULTADOS
print("\n--- Resultados do GridSearchCV com Engenharia e Seleção de Features ---")
print("Melhores parâmetros encontrados:")
print(grid_search_final.best_params_)

# Avaliação do melhor modelo
print("\n--- Avaliação Final do Melhor Modelo ---")
best_model = grid_search_final.best_estimator_
y_pred = best_model.predict(X_test_eng)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Para inspecionar quais features foram selecionadas
# Acessamos a etapa de seleção do melhor pipeline treinado
selected_features_mask = grid_search_final.best_estimator_.named_steps['feature_selection'].get_support()
# Obter nomes das features após o pré-processamento
feature_names_after_preprocessing = grid_search_final.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
selected_features = np.array(feature_names_after_preprocessing)[selected_features_mask]

print(f"\nNúmero de features selecionadas: {len(selected_features)} de {len(feature_names_after_preprocessing)}")
print("Features Selecionadas pelo Modelo:")
print(list(selected_features))