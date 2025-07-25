# models/base.py - VERSÃO ATUALIZADA PARA CONFIGURAÇÕES GLOBAIS
'''
Classes base para modelos hipsométricos e volumétricos
NOVO: Suporte aprimorado para configurações de otimização
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

        # NOVO: Configurações de otimização
        self.max_iter = 5000
        self.tolerancia = 0.01

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

    def configurar_otimizacao(self, max_iter=None, tolerancia=None):
        """
        NOVO: Configura parâmetros de otimização

        Args:
            max_iter: Máximo de iterações
            tolerancia: Tolerância para convergência
        """
        if max_iter is not None:
            self.max_iter = max_iter
        if tolerancia is not None:
            self.tolerancia = tolerancia


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
    '''Classe base para modelos não-lineares - VERSÃO APRIMORADA'''

    def __init__(self, nome, funcao, params_iniciais):
        super().__init__(nome)
        self.funcao = funcao
        self.params_iniciais = params_iniciais
        self.parametros = None
        self.info_convergencia = {}

    def ajustar(self, X, y):
        '''Ajusta modelo não-linear com curve_fit - VERSÃO APRIMORADA'''
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

            # NOVO: Configurar parâmetros de otimização
            kwargs_otimizacao = {
                'p0': self.params_iniciais,
                'maxfev': self.max_iter
            }

            # NOVO: Aplicar bounds se necessário para melhor convergência
            if self.nome in ['Chapman', 'Weibull', 'Mononuclear']:
                # Definir bounds razoáveis para modelos hipsométricos
                bounds_inf = [10.0, 0.001, 0.1]  # Limites inferiores
                bounds_sup = [60.0, 1.0, 5.0]  # Limites superiores

                # Ajustar bounds baseado no modelo específico
                if self.nome == 'Mononuclear':
                    bounds_inf[1] = 0.1  # b deve ser >= 0.1
                    bounds_sup[1] = 2.0  # b deve ser <= 2.0

                kwargs_otimizacao['bounds'] = (bounds_inf, bounds_sup)

            # Ajustar com configurações otimizadas
            self.parametros, pcov = curve_fit(
                self.funcao,
                X_clean,
                y_clean,
                **kwargs_otimizacao
            )

            # NOVO: Calcular informações de convergência
            self.info_convergencia = {
                'parametros_iniciais': self.params_iniciais.copy(),
                'parametros_finais': self.parametros.copy(),
                'covariancia': pcov,
                'dados_usados': len(X_clean),
                'convergiu': True
            }

            # NOVO: Verificar qualidade da convergência
            if np.any(np.diag(pcov) > 1e6):  # Covariância muito alta indica problemas
                self.info_convergencia['convergiu'] = False
                print(f"⚠️ {self.nome}: Possível problema de convergência (covariância alta)")

            self.ajustado = True
            return True

        except Exception as e:
            self.ajustado = False
            self.info_convergencia = {
                'erro': str(e),
                'convergiu': False,
                'parametros_iniciais': self.params_iniciais.copy()
            }
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

    def obter_info_convergencia(self):
        """
        NOVO: Retorna informações detalhadas sobre a convergência

        Returns:
            dict: Informações sobre convergência e parâmetros
        """
        return self.info_convergencia

    def validar_parametros(self):
        """
        NOVO: Valida se os parâmetros finais são razoáveis

        Returns:
            dict: Status da validação
        """
        if not self.ajustado:
            return {'valido': False, 'motivo': 'Modelo não ajustado'}

        if not self.info_convergencia.get('convergiu', False):
            return {'valido': False, 'motivo': 'Problemas de convergência'}

        # Verificações específicas por tipo de modelo
        if self.nome in ['Chapman', 'Weibull', 'Mononuclear']:
            # Para modelos hipsométricos, verificar se altura assintótica é razoável
            altura_assintotica = self.parametros[0]
            if altura_assintotica < 5 or altura_assintotica > 80:
                return {
                    'valido': False,
                    'motivo': f'Altura assintótica irrealística: {altura_assintotica:.2f}m'
                }

        return {'valido': True, 'motivo': 'Parâmetros válidos'}


def calcular_r2_generalizado(y_obs, y_pred):
    '''Calcula R² generalizado para modelos transformados'''
    return 1 - (np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2))


def ajustar_modelo_seguro(X, y, nome_modelo, tipo='linear', config=None):
    '''
    Ajusta modelo com tratamento completo de erros - VERSÃO ATUALIZADA

    Args:
        X: Variáveis independentes
        y: Variável dependente
        nome_modelo: Nome do modelo
        tipo: 'linear' ou 'nao_linear'
        config: Configurações de otimização (NOVO)

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

        # NOVO: Aplicar configurações se disponíveis (para modelos lineares isso não muda muito)
        if config and tipo == 'linear':
            # Podemos adicionar regularização ou outros parâmetros no futuro
            pass

        modelo.fit(X_clean, y_clean)

        # Predições para todo dataset (preenchendo NaN)
        X_pred = X.fillna(X.mean())
        y_pred = modelo.predict(X_pred)

        r2 = r2_score(y.fillna(y.mean()), y_pred)
        rmse = np.sqrt(mean_squared_error(y.fillna(y.mean()), y_pred))

        resultado = {
            'modelo': modelo,
            'r2': r2,
            'rmse': rmse,
            'y_pred': y_pred,
            'sucesso': True
        }

        # NOVO: Adicionar informações de configuração se disponível
        if config:
            resultado['config_aplicada'] = config

        return resultado

    except Exception as e:
        return {
            'erro': str(e),
            'sucesso': False
        }


def aplicar_configuracoes_modelo(modelo, config):
    """
    NOVA: Aplica configurações globais a um modelo específico

    Args:
        modelo: Instância do modelo
        config: Configurações globais
    """
    if hasattr(modelo, 'configurar_otimizacao'):
        max_iter = config.get('max_iteracoes', 5000)
        tolerancia = config.get('tolerancia_ajuste', 0.01)
        modelo.configurar_otimizacao(max_iter, tolerancia)


def validar_resultado_modelo(resultado, modelo_nome, config=None):
    """
    NOVA: Valida resultado de um modelo baseado nas configurações

    Args:
        resultado: Resultado do ajuste do modelo
        modelo_nome: Nome do modelo
        config: Configurações para validação

    Returns:
        dict: Status da validação
    """
    if not resultado.get('sucesso', False):
        return {'valido': False, 'motivo': 'Modelo não foi ajustado'}

    # Validação básica de R²
    r2 = resultado.get('r2', 0)
    r2_minimo = 0.1  # Padrão muito baixo

    if config:
        # Calcular R² mínimo baseado na tolerância
        tolerancia = config.get('tolerancia_ajuste', 0.01)
        r2_minimo = max(0.1, 0.5 - tolerancia * 10)  # Entre 0.1 e 0.5

    if r2 < r2_minimo:
        return {
            'valido': False,
            'motivo': f'R² muito baixo: {r2:.3f} < {r2_minimo:.3f}'
        }

    # Validação de RMSE (não pode ser infinito ou NaN)
    rmse = resultado.get('rmse', float('inf'))
    if not np.isfinite(rmse):
        return {
            'valido': False,
            'motivo': 'RMSE inválido (infinito ou NaN)'
        }

    return {'valido': True, 'motivo': 'Modelo válido'}


def gerar_relatorio_tecnico_modelo(modelo, resultado):
    """
    NOVA: Gera relatório técnico detalhado de um modelo

    Args:
        modelo: Instância do modelo
        resultado: Resultado do ajuste

    Returns:
        str: Relatório técnico
    """
    relatorio = f"## RELATÓRIO TÉCNICO - {modelo.nome}\n\n"

    # Informações básicas
    relatorio += f"**Tipo**: {'Não-linear' if isinstance(modelo, ModeloNaoLinear) else 'Linear'}\n"
    relatorio += f"**Status**: {'Ajustado' if modelo.ajustado else 'Não ajustado'}\n"
    relatorio += f"**R²**: {resultado.get('r2', 'N/A'):.4f}\n"
    relatorio += f"**RMSE**: {resultado.get('rmse', 'N/A'):.4f}\n\n"

    # Informações específicas para modelos não-lineares
    if isinstance(modelo, ModeloNaoLinear) and hasattr(modelo, 'info_convergencia'):
        info_conv = modelo.info_convergencia
        relatorio += "### Convergência\n"
        relatorio += f"**Convergiu**: {'Sim' if info_conv.get('convergiu', False) else 'Não'}\n"
        relatorio += f"**Dados utilizados**: {info_conv.get('dados_usados', 'N/A')}\n"

        if 'parametros_iniciais' in info_conv and 'parametros_finais' in info_conv:
            relatorio += "\n**Parâmetros:**\n"
            for i, (inicial, final) in enumerate(zip(info_conv['parametros_iniciais'], info_conv['parametros_finais'])):
                relatorio += f"- Parâmetro {i + 1}: {inicial:.4f} → {final:.4f}\n"

        # Validação dos parâmetros
        validacao = modelo.validar_parametros()
        relatorio += f"\n**Validação**: {validacao['motivo']}\n"

    # Configurações aplicadas
    if hasattr(modelo, 'max_iter') and hasattr(modelo, 'tolerancia'):
        relatorio += f"\n### Configurações\n"
        relatorio += f"**Máx. iterações**: {modelo.max_iter}\n"
        relatorio += f"**Tolerância**: {modelo.tolerancia}\n"

    return relatorio