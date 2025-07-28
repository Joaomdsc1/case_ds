import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Carregar o dataset
try:
    df = pd.read_csv('teste_lifecycledatascience.csv', sep=';')
except FileNotFoundError:
    print("Arquivo 'teste_lifecycledatascience.csv' não encontrado.")
    exit()


# 1. Limpeza e Pré-processamento dos Dados
# Remover colunas vazias que podem ter sido criadas por erro de formatação no CSV
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how='all')

# Converter colunas monetárias para formato numérico
for col in ['Gasto_Crédito', 'Dinheiro_Investido']:
    if df[col].dtype == 'object':
        df[col] = df[col].replace({'R\$ ': '', '\.': ''}, regex=True).str.replace(',', '.').astype(float)

# 2. Engenharia de Features: Criar a variável alvo 'segmento'
# Definir clientes de 'alto valor' como aqueles no quartil superior de gastos com cartão de crédito
limite_alto_valor = df['Gasto_Crédito'].quantile(0.75)
df['segmento'] = np.where(df['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')

# 3. Preparação para o Modelo
# Separar as features (X) e a variável alvo (y)
X = df.drop(['Cliente', 'segmento', 'Gasto_Crédito'], axis=1) # Gasto_Crédito é removido para evitar vazamento de dados
y = df['segmento']

# Identificar colunas categóricas e numéricas
categorical_features = ['Gênero']
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# 4. Criação do Pipeline de Pré-processamento
# Criar um transformador para as variáveis numéricas (normalização) e categóricas (one-hot encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 5. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Criação e Treinamento do Modelo
# Usar um pipeline para encadear o pré-processamento e o modelo de Regressão Logística
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42))])

# Treinar o modelo
model.fit(X_train, y_train)

# 7. Avaliação do Modelo
# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)

print("--- Avaliação do Modelo de Classificação ---")
print(f"Acurácia do modelo: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Para fins de análise da persona, podemos prever os segmentos para todo o conjunto de dados
df['segmento_predito'] = model.predict(X)

print("\n--- Análise da Persona de Alto Valor ---")
# Análise descritiva do segmento 'alto_valor'
persona_alto_valor = df[df['segmento_predito'] == 'alto_valor'].describe()
print(persona_alto_valor)