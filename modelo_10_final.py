import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Carregar o dataset
try:
    df = pd.read_csv('teste_lifecycledatascience.csv', sep=';')
except FileNotFoundError:
    print("Arquivo 'teste_lifecycledatascience.csv' não encontrado.")
    exit()

# 1. Limpeza e Pré-processamento Inicial
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how='all')
for col in ['Gasto_Crédito', 'Dinheiro_Investido']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].replace({'R\$ ': '', '\.': ''}, regex=True).str.replace(',', '.').astype(float)

# 2. Engenharia de Features
print("Iniciando engenharia de features...")
epsilon = 1e-6 
df['Proporcao_Gasto_Investimento'] = df['Gasto_Crédito'] / (df['Dinheiro_Investido'] + epsilon)
df['Score_Idade_Investimento'] = df['Idade'] * df['Dinheiro_Investido']
print("Novas features criadas: 'Proporcao_Gasto_Investimento', 'Score_Idade_Investimento'")


# 3. Prevenção de Data Leak (Estrutura Mantida)
X = df.drop('Cliente', axis=1)
y_placeholder = df['Cliente']
stratify_proxy = df['Gasto_Crédito'] > df['Gasto_Crédito'].median()
X_temp, X_test_full, _, _ = train_test_split(
    X, y_placeholder, test_size=0.2, random_state=42, stratify=stratify_proxy
)
limite_alto_valor = X_temp['Gasto_Crédito'].quantile(0.75)
y_temp = np.where(X_temp['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')
y_test = np.where(X_test_full['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')
X_temp = X_temp.drop('Gasto_Crédito', axis=1)
X_test = X_test_full.drop('Gasto_Crédito', axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# 4. Identificação de features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nFeatures numéricas atualizadas: {numeric_features}")

# 5. Pipeline de Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# 6. Pipeline Principal
USAR_SMOTE = True
pipeline_steps = [('preprocessor', preprocessor)]
if USAR_SMOTE:
    print("\nSMOTE está ATIVADO.")
    pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=3)))
else:
    print("\nSMOTE está DESATIVADO.")
pipeline_steps.append(('classifier', lgb.LGBMClassifier(
    random_state=42, objective='binary', metric='binary_logloss', verbosity=-1, force_col_wise=True
)))
pipeline = Pipeline(steps=pipeline_steps)

# 7. Grid de Hiperparâmetros para Combater Overfitting
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__num_leaves': [7, 10, 15],
    'classifier__min_child_samples': [20, 30, 40],
    'classifier__subsample': [0.8, 0.9],
    'classifier__colsample_bytree': [0.8, 0.9],
    'classifier__reg_alpha': [0.01, 0.1, 1],
    'classifier__reg_lambda': [0.1, 1, 5]
}

# 8. GridSearch com Validação Cruzada
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv_strategy, n_jobs=-1, scoring='roc_auc', verbose=1, return_train_score=True
)
print("\nIniciando GridSearchCV com novo grid focado em regularização...")
grid_search.fit(X_train, y_train)

# 9. Análise dos resultados
print("\n" + "="*60)
print("RESULTADOS DO NOVO GRIDSEARCHCV")
print("="*60)
print("Melhores parâmetros:")
print(grid_search.best_params_)
print(f"\nMelhor score CV (AUC): {grid_search.best_score_:.4f}")

# 10. Avaliação no conjunto de validação
print("\n" + "="*60)
print("AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO")
print("="*60)
y_val_encoded = (y_val == 'alto_valor').astype(int)
y_val_pred = grid_search.predict(X_val)
# ==============================================================================
y_val_proba = grid_search.predict_proba(X_val)[:, 0]
# ==============================================================================
print("Relatório de Classificação (Validação):")
print(classification_report(y_val, y_val_pred))
print(f"AUC-ROC (Validação): {roc_auc_score(y_val_encoded, y_val_proba):.4f}")

# 11. Avaliação final no conjunto de teste
print("\n" + "="*60)
print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
print("="*60)
y_test_encoded = (y_test == 'alto_valor').astype(int)
y_test_pred = grid_search.predict(X_test)
# ==============================================================================
y_test_proba = grid_search.predict_proba(X_test)[:, 0]
# ==============================================================================
print("Relatório de Classificação (Teste):")
print(classification_report(y_test, y_test_pred))
print(f"AUC-ROC (Teste): {roc_auc_score(y_test_encoded, y_test_proba):.4f}")


# 12. Diagnóstico de overfitting
print("\n" + "="*60)
print("DIAGNÓSTICO DE OVERFITTING")
print("="*60)
y_train_encoded = (y_train == 'alto_valor').astype(int)
# ==============================================================================
train_proba = grid_search.predict_proba(X_train)[:, 0]
# ==============================================================================
train_auc = roc_auc_score(y_train_encoded, train_proba)
val_auc = roc_auc_score(y_val_encoded, y_val_proba)
test_auc = roc_auc_score(y_test_encoded, y_test_proba)

print(f"AUC Treino: {train_auc:.4f}")
print(f"AUC Validação: {val_auc:.4f}")
print(f"AUC Teste: {test_auc:.4f}")

diff_train_val = abs(train_auc - val_auc)
print(f"\nDiferença (Absoluta) AUC Treino-Validação: {diff_train_val:.4f}")

if diff_train_val > 0.15:
    print("⚠️  ATENÇÃO: Overfitting ainda pode estar presente.")
else:
    print("✅ Modelo parece bem generalizado.")