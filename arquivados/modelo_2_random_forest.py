import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
try:
    df = pd.read_csv('teste_lifecycledatascience.csv', sep=';')
except FileNotFoundError:
    print("Arquivo 'teste_lifecycledatascience.csv' não encontrado.")
    exit()

# 1. Limpeza e Pré-processamento
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
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Criação e Treinamento do Modelo com Random Forest
# Usamos o RandomForestClassifier com o parâmetro class_weight='balanced' para lidar com o desbalanceamento
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

# Treinar o modelo
model.fit(X_train, y_train)

# 7. Avaliação do Modelo
y_pred = model.predict(X_test)
print("--- Avaliação do Modelo Random Forest (com balanceamento) ---")
print(f"Acurácia do modelo: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# 8. Análise da Importância das Features
# Extrair os nomes das features após o OneHotEncoding
ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_feature_names)

# Extrair as importâncias
importances = model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Importância das Features para o Modelo ---")
print(feature_importance_df)

# Gerar gráfico de importância das features para visualização
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importância de Cada Feature para Definir o Segmento')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nGráfico 'feature_importance.png' gerado com sucesso!")
