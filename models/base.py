# models/base.py
'''
Classes base para modelos hipsométricos e volumétricos
'''

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit


class ModeloBase(ABC):
    '''Classe base abstrata para todos os modelos'''

    def __init__(self, nome):
        self.nome = nome
        self.modelo = None
        self.r2 = None
        self.rmse = None
        self.predicoes = None
        self.ajustado = False

    @abstractmethod
    def preparar_dados(self, df):
        '''Prepara os dados para o modelo específico'''
        pass

    @abstractmethod
    def ajustar(self, X, y):
        '''Ajusta o modelo aos dados'''
        pass

    @abstractmethod
    def predizer(self, X):
        '''Faz predições com o modelo ajustado'''
        pass

    def avaliar(self, y_obs, y_pred):
        '''Avalia a qualidade do modelo'''
        self.r2 = r2_score(y_obs, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
        return {'r2': self.r2, 'rmse': self.rmse}


class ModeloLinear(ModeloBase):
    '''Classe base para modelos lineares'''

    def __init__(self, nome):
        super().__init__(nome)
        self.modelo = LinearRegression()

    def ajustar(self, X, y):
        '''Ajusta modelo linear com tratamento de NaN'''
        try:
            # Verificar e limpar NaN
            if X.isna().any().any() or y.isna().any():
                mask_validos = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask_validos]
                y_clean = y[mask_validos]

                if len(X_clean) < 10:
                    raise ValueError(f"Poucos dados válidos: {len(X_clean)}")
            else:
                X_clean = X
                y_clean = y

            # Ajustar modelo
            self.modelo.fit(X_clean, y_clean)
            self.ajustado = True

            return True

        except Exception as e:
            self.ajustado = False
            return False

    def predizer(self, X):
        '''Faz predições preenchendo NaN com médias'''
        if not self.ajustado:
            raise ValueError("Modelo não foi ajustado")

        # Preencher NaN com médias
        X_pred = X.fillna(X.mean())
        return self.modelo.predict(X_pred)


class ModeloNaoLinear(ModeloBase):
    '''Classe base para modelos não-lineares'''

    def __init__(self, nome, funcao, params_iniciais):
        super().__init__(nome)
        self.funcao = funcao
        self.params_iniciais = params_iniciais
        self.parametros = None

    def ajustar(self, X, y):
        '''Ajusta modelo não-linear com curve_fit'''
        try:
            # Limpar dados
            if isinstance(X, pd.DataFrame):
                mask_validos = X.notna().all(axis=1) & y.notna()
            else:
                mask_validos = pd.notna(X) & pd.notna(y)

            if mask_validos.sum() < 20:
                raise ValueError("Poucos dados válidos para modelo não-linear")

            if isinstance(X, pd.DataFrame):
                X_clean = X[mask_validos].iloc[:, 0]  # Assumir primeira coluna
            else:
                X_clean = X[mask_validos]
            y_clean = y[mask_validos]

            # Ajustar
            self.parametros, _ = curve_fit(
                self.funcao,
                X_clean,
                y_clean,
                p0=self.params_iniciais,
                maxfev=5000
            )
            self.ajustado = True
            return True

        except Exception as e:
            self.ajustado = False
            return False

    def predizer(self, X):
        '''Faz predições com modelo não-linear'''
        if not self.ajustado:
            raise ValueError("Modelo não foi ajustado")

        if isinstance(X, pd.DataFrame):
            X_pred = X.fillna(X.mean()).iloc[:, 0]
        else:
            X_pred = pd.Series(X).fillna(pd.Series(X).mean())

        return self.funcao(X_pred, *self.parametros)


def calcular_r2_generalizado(y_obs, y_pred):
    '''Calcula R² generalizado para modelos transformados'''
    return 1 - (np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2))


def ajustar_modelo_seguro(X, y, nome_modelo, tipo='linear'):
    '''
    Ajusta modelo com tratamento completo de erros

    Args:
        X: Variáveis independentes
        y: Variável dependente  
        nome_modelo: Nome do modelo
        tipo: 'linear' ou 'nao_linear'

    Returns:
        dict: Resultado do ajuste ou None se falhou
    '''
    try:
        # Verificar NaN
        if X.isna().any().any() or y.isna().any():
            mask_validos = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask_validos]
            y_clean = y[mask_validos]

            if len(X_clean) < 10:
                raise ValueError(f"Poucos dados válidos: {len(X_clean)}")
        else:
            X_clean = X
            y_clean = y

        # Ajustar modelo
        modelo = LinearRegression()
        modelo.fit(X_clean, y_clean)

        # Predições para todo dataset (preenchendo NaN)
        X_pred = X.fillna(X.mean())
        y_pred = modelo.predict(X_pred)

        r2 = r2_score(y.fillna(y.mean()), y_pred)
        rmse = np.sqrt(mean_squared_error(y.fillna(y.mean()), y_pred))

        return {
            'modelo': modelo,
            'r2': r2,
            'rmse': rmse,
            'y_pred': y_pred,
            'sucesso': True
        }

    except Exception as e:
        return {
            'erro': str(e),
            'sucesso': False
        }