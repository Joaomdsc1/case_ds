import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, make_scorer, recall_score)
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Classe para processamento e limpeza de dados"""
    
    def __init__(self):
        self.scaler = None
        self.encoder = None
    
    def load_data(self, filepath: str, separator: str = ';') -> pd.DataFrame:
        """Carrega dados com tratamento de erro"""
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                # Cria um arquivo de exemplo se ele não existir
                logger.warning(f"Arquivo '{filepath}' não encontrado. Criando um arquivo de exemplo.")
                data = {
                    'Cliente': range(1000),
                    'Gasto_Crédito': np.random.uniform(100, 10000, 1000),
                    'Dinheiro_Investido': np.random.uniform(500, 50000, 1000),
                    'Idade': np.random.randint(18, 70, 1000),
                    'Plano_Ativo': np.random.choice(['Sim', 'Não'], 1000, p=[0.7, 0.3])
                }
                df = pd.DataFrame(data)
                df.to_csv(filepath, sep=separator, index=False)
            
            df = pd.read_csv(filepath, sep=separator)
            logger.info(f"Dataset carregado com sucesso: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e processa os dados"""
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')
        
        monetary_cols = ['Gasto_Crédito', 'Dinheiro_Investido']
        for col in monetary_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = self._clean_monetary_column(df[col])
        
        logger.info(f"Dados limpos: {df.shape}")
        return df
    
    def _clean_monetary_column(self, series: pd.Series) -> pd.Series:
        cleaned = (series.astype(str)
                  .str.replace(r'R\$\s*', '', regex=True)
                  .str.replace(r'\.(?=\d{3})', '', regex=True)
                  .str.replace(',', '.')
                  .str.replace(r'[^\d.]', '', regex=True))
        return pd.to_numeric(cleaned, errors='coerce')

class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'Gasto_Crédito' in df.columns:
            limite_alto_valor = df['Gasto_Crédito'].quantile(0.75)
            df['segmento'] = np.where(df['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')
        
        if 'Gasto_Crédito' in df.columns and 'Dinheiro_Investido' in df.columns:
            df['razao_gasto_investimento'] = df['Gasto_Crédito'] / (df['Dinheiro_Investido'] + 1)
            df['total_movimentacao'] = df['Gasto_Crédito'] + df['Dinheiro_Investido']
            df['perfil_financeiro'] = pd.cut(df['total_movimentacao'], 
                                           bins=3, 
                                           labels=['conservador', 'moderado', 'agressivo'])
        
        logger.info("Features criadas com sucesso")
        return df

class ModelEvaluator:
    """Classe para avaliação de modelos"""
    
    def evaluate_model(self, y_test: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, Any]:
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        logger.info(f"Avaliação do modelo {model_name} concluída")
        return results
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name: str = "Model"):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Valores Reais')
        plt.xlabel('Predições')
        plt.show()

def main():
    """Função principal focada em maximizar o recall"""
    
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer() 
    evaluator = ModelEvaluator()

    try:
        df = data_processor.load_data('teste_lifecycledatascience.csv')
        df = data_processor.clean_data(df)
        df = feature_engineer.create_features(df)
        
        if 'segmento' not in df.columns:
            raise ValueError("Coluna 'segmento' não foi criada.")
            
    except Exception as e:
        logger.error(f"Erro no processamento inicial: {e}")
        return
    
    cols_to_remove = ['Cliente', 'segmento', 'Gasto_Crédito']
    X = df.drop([col for col in cols_to_remove if col in df.columns], axis=1)
    y = df['segmento']
    
    logger.info(f"Distribuição das classes:\n{y.value_counts(normalize=True)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('sampler', SMOTE(random_state=42)), # Placeholder, será substituído pelo GridSearchCV
        ('classifier', lgb.LGBMClassifier(
            random_state=42,
            objective='binary',
            verbose=-1
        ))
    ])
    
    param_grid = {
        'sampler': [SMOTE(random_state=42), ADASYN(random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [31, 50],
        'classifier__class_weight': ['balanced', None] # Testar com e sem pesos de classe
    }
    
    # Queremos maximizar o recall para a classe 'alto_valor'
    recall_scorer = make_scorer(recall_score, pos_label='alto_valor')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv,
        n_jobs=-1, 
        scoring=recall_scorer, 
        verbose=1
    )
    
    logger.info("Iniciando busca de hiperparâmetros para maximizar o RECALL...")
    grid_search.fit(X_train, y_train)
    
    logger.info("=== RESULTADOS FINAIS DA BUSCA ===")
    logger.info(f"Melhor score de RECALL (CV): {grid_search.best_score_:.4f}")
    logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # --- Avaliação com o limiar padrão (0.5) ---
    y_pred_default = best_model.predict(X_test)
    print("\n" + "="*50)
    print("RELATÓRIO DE CLASSIFICAÇÃO (LIMIAR PADRÃO 0.5)")
    print("="*50)
    print(classification_report(y_test, y_pred_default))
    evaluator.plot_confusion_matrix(y_test, y_pred_default, "LightGBM - Limiar Padrão")
    
    # --- Avaliação com o limiar ajustado para aumentar o RECALL ---
    print("\n" + "="*50)
    print("AJUSTE DE LIMIAR PARA AUMENTAR RECALL")
    print("="*50)

    # Obter as probabilidades da classe positiva ('alto_valor')
    # Descobrir o índice da classe 'alto_valor'
    positive_class_index = np.where(best_model.classes_ == 'alto_valor')[0][0]
    y_pred_proba = best_model.predict_proba(X_test)[:, positive_class_index]
    
    # Definir um novo limiar mais baixo para capturar mais positivos
    novo_limiar = 0.3
    logger.info(f"Aplicando novo limiar de decisão: {novo_limiar}")

    # Aplicar o novo limiar para gerar as predições
    y_pred_ajustado = np.where(y_pred_proba >= novo_limiar, 'alto_valor', 'padrao')

    print(f"\nRELATÓRIO DE CLASSIFICAÇÃO (LIMIAR AJUSTADO {novo_limiar})")
    print("-" * 50)
    print(classification_report(y_test, y_pred_ajustado))
    evaluator.plot_confusion_matrix(y_test, y_pred_ajustado, f"LightGBM - Limiar {novo_limiar}")

if __name__ == "__main__":
    main()