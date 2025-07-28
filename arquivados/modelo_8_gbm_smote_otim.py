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
                           roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
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
                raise FileNotFoundError(f"Arquivo '{filepath}' não encontrado.")
            
            df = pd.read_csv(filepath, sep=separator)
            logger.info(f"Dataset carregado com sucesso: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e processa os dados"""
        # Remove colunas unnamed
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Remove colunas totalmente vazias
        df = df.dropna(axis=1, how='all')
        
        # Processa colunas monetárias de forma mais robusta
        monetary_cols = ['Gasto_Crédito', 'Dinheiro_Investido']
        for col in monetary_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = self._clean_monetary_column(df[col])
        
        logger.info(f"Dados limpos: {df.shape}")
        return df
    
    def _clean_monetary_column(self, series: pd.Series) -> pd.Series:
        """Limpa colunas monetárias"""
        # Remove símbolos monetários e converte para float
        cleaned = (series.astype(str)
                  .str.replace(r'R\$\s*', '', regex=True)
                  .str.replace(r'\.(?=\d{3})', '', regex=True)  # Remove pontos de milhares
                  .str.replace(',', '.')  # Converte vírgula decimal para ponto
                  .str.replace(r'[^\d.]', '', regex=True))  # Remove outros caracteres
        
        return pd.to_numeric(cleaned, errors='coerce')

class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria novas features"""
        df = df.copy()
        
        # Feature original
        if 'Gasto_Crédito' in df.columns:
            limite_alto_valor = df['Gasto_Crédito'].quantile(0.75)
            df['segmento'] = np.where(df['Gasto_Crédito'] >= limite_alto_valor, 'alto_valor', 'padrao')
        
        # Novas features adicionais
        if 'Gasto_Crédito' in df.columns and 'Dinheiro_Investido' in df.columns:
            # Razão entre gasto e investimento
            df['razao_gasto_investimento'] = df['Gasto_Crédito'] / (df['Dinheiro_Investido'] + 1)
            
            # Total de movimentação financeira
            df['total_movimentacao'] = df['Gasto_Crédito'] + df['Dinheiro_Investido']
            
            # Categoria de perfil financeiro
            df['perfil_financeiro'] = pd.cut(df['total_movimentacao'], 
                                           bins=3, 
                                           labels=['conservador', 'moderado', 'agressivo'])
        
        logger.info("Features criadas com sucesso")
        return df

class ModelEvaluator:
    """Classe para avaliação de modelos"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, Any]:
        """Avalia o modelo de forma abrangente"""
        
        # Predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Métricas
        results = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred
        }
        
        # AUC-ROC se probabilidades disponíveis
        if y_pred_proba is not None:
            # Converter labels para binário se necessário
            if len(np.unique(y_test)) == 2:
                le = LabelEncoder()
                y_test_encoded = le.fit_transform(y_test)
                results['auc_roc'] = roc_auc_score(y_test_encoded, y_pred_proba)
        
        self.results[model_name] = results
        logger.info(f"Avaliação do modelo {model_name} concluída")
        
        return results
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name: str = "Model"):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Valores Reais')
        plt.xlabel('Predições')
        plt.show()

def main():
    """Função principal"""
    
    # 1. Inicialização das classes
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer() 
    evaluator = ModelEvaluator()
    
    # 2. Carregamento e processamento dos dados
    try:
        df = data_processor.load_data('teste_lifecycledatascience.csv')
        df = data_processor.clean_data(df)
        df = feature_engineer.create_features(df)
        
        # Verificar se a coluna target foi criada
        if 'segmento' not in df.columns:
            raise ValueError("Coluna 'segmento' não foi criada. Verifique os dados.")
            
    except Exception as e:
        logger.error(f"Erro no processamento inicial: {e}")
        return
    
    # 3. Preparação para modelagem
    # Features a serem removidas
    cols_to_remove = ['Cliente', 'segmento']
    if 'Gasto_Crédito' in df.columns:
        cols_to_remove.append('Gasto_Crédito')
    
    X = df.drop([col for col in cols_to_remove if col in df.columns], axis=1)
    y = df['segmento']
    
    # Verificar balanceamento das classes
    logger.info(f"Distribuição das classes:\n{y.value_counts()}")
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Pipeline de pré-processamento mais robusto
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Features categóricas: {categorical_features}")
    logger.info(f"Features numéricas: {numeric_features}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'  # Manter outras colunas
    )
    
    # 5. Pipeline principal
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=min(5, len(y_train[y_train == y_train.value_counts().idxmin()]) - 1))),
        ('classifier', lgb.LGBMClassifier(
            random_state=42,
            objective='binary',
            verbose=-1 
        ))
    ])
    
    # 6. Grid de hiperparâmetros expandido
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__num_leaves': [20, 31, 50],
        'classifier__max_depth': [-1, 5, 10],
        'classifier__min_data_in_leaf': [20, 30, 40],
    }
    
    # 7. GridSearchCV com validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv,
        n_jobs=-1, 
        scoring='f1_weighted',
        verbose=1, 
        return_train_score=True
    )
    
    # 8. Treinamento
    logger.info("Iniciando busca de hiperparâmetros...")
    grid_search.fit(X_train, y_train)
    
    # 9. Resultados detalhados
    logger.info("=== RESULTADOS FINAIS ===")
    logger.info(f"Melhor score CV: {grid_search.best_score_:.4f}")
    logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
    
    # 10. Avaliação final
    best_model = grid_search.best_estimator_
    results = evaluator.evaluate_model(best_model, X_test, y_test, "LightGBM_Best")
    
    # Relatório detalhado
    print("\n" + "="*50)
    print("RELATÓRIO FINAL DE CLASSIFICAÇÃO")
    print("="*50)
    print(classification_report(y_test, results['predictions']))
    
    # Plot da matriz de confusão
    evaluator.plot_confusion_matrix(y_test, results['predictions'], "LightGBM")
    
    # Feature importance se disponível
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        try:
            importances = best_model.named_steps['classifier'].feature_importances_
            
            # Obter nomes das features após o preprocessamento
            if hasattr(best_model.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            else:
                # Fallback: criar nomes genéricos
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Garantir que os arrays tenham o mesmo tamanho
            min_length = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_length]
            importances = importances[:min_length]
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 features mais importantes:")
            print(feature_importance_df.head(10))
            
        except Exception as e:
            logger.warning(f"Não foi possível gerar feature importance: {e}")
            logger.info(f"Número de features: {len(importances) if 'importances' in locals() else 'N/A'}")

if __name__ == "__main__":
    main()