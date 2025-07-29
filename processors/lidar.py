# processors/lidar.py
'''
Processamento e integração de dados LiDAR com inventário florestal
Integra métricas extraídas do script R com dados de campo
'''

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import (
    COLUNAS_LIDAR_OBRIGATORIAS,
    NOMES_ALTERNATIVOS_LIDAR,
    CONFIGURACOES_LIDAR,
    THRESHOLDS_ALERTAS_LIDAR,
    CLASSIFICACAO_CALIBRACAO_LIDAR,
    CORES_LIDAR
)


def processar_dados_lidar(arquivo_lidar):
    """
    Processa arquivo de métricas LiDAR gerado pelo script R

    Args:
        arquivo_lidar: Arquivo CSV/Excel com métricas das parcelas

    Returns:
        DataFrame com métricas LiDAR processadas ou None se erro
    """
    try:
        from utils.arquivo_handler import carregar_arquivo

        # Carregar arquivo
        df_lidar = carregar_arquivo(arquivo_lidar)
        if df_lidar is None:
            st.error("❌ Não foi possível carregar arquivo LiDAR")
            return None

        # Validar estrutura básica
        validacao = validar_estrutura_lidar(df_lidar)
        if not validacao['valido']:
            st.error("❌ Estrutura de dados LiDAR inválida:")
            for erro in validacao['erros']:
                st.error(f"• {erro}")
            return None

        # Padronizar nomes das colunas
        df_lidar = padronizar_colunas_lidar(df_lidar)

        # Limpar e validar dados
        df_lidar = limpar_dados_lidar(df_lidar)

        # Calcular métricas derivadas
        df_lidar = calcular_metricas_derivadas_lidar(df_lidar)

        st.success(f"✅ Dados LiDAR processados: {len(df_lidar)} parcelas")

        return df_lidar

    except Exception as e:
        st.error(f"❌ Erro ao processar dados LiDAR: {e}")
        return None


def validar_estrutura_lidar(df_lidar):
    """
    Valida estrutura básica dos dados LiDAR

    Args:
        df_lidar: DataFrame com dados LiDAR

    Returns:
        dict: Resultado da validação
    """
    validacao = {'valido': True, 'erros': [], 'alertas': []}

    # Verificar se DataFrame não está vazio
    if len(df_lidar) == 0:
        validacao['erros'].append("Arquivo LiDAR vazio")
        validacao['valido'] = False
        return validacao

    # Verificar colunas obrigatórias
    colunas_faltantes = []
    for col in COLUNAS_LIDAR_OBRIGATORIAS:
        if col not in df_lidar.columns:
            # Tentar encontrar equivalentes
            encontrada = False
            for col_orig in df_lidar.columns:
                if col.lower() in col_orig.lower():
                    encontrada = True
                    break
            if not encontrada:
                colunas_faltantes.append(col)

    if colunas_faltantes:
        validacao['erros'].append(f"Colunas obrigatórias faltantes: {colunas_faltantes}")
        validacao['valido'] = False

    # Verificar se há pelo menos uma métrica LiDAR
    metricas_encontradas = 0
    for metrica_padrao, aliases in NOMES_ALTERNATIVOS_LIDAR.items():
        for alias in aliases:
            if alias in df_lidar.columns:
                metricas_encontradas += 1
                break

    if metricas_encontradas == 0:
        validacao['erros'].append("Nenhuma métrica LiDAR reconhecida encontrada")
        validacao['valido'] = False
    elif metricas_encontradas < 3:
        validacao['alertas'].append(f"Poucas métricas LiDAR encontradas: {metricas_encontradas}")

    return validacao


def padronizar_colunas_lidar(df_lidar):
    """
    Padroniza nomes das colunas baseado nos aliases conhecidos

    Args:
        df_lidar: DataFrame com dados LiDAR

    Returns:
        DataFrame com colunas padronizadas
    """
    df_padronizado = df_lidar.copy()

    # Mapear colunas para nomes padrão
    mapeamento = {}

    for nome_padrao, aliases in NOMES_ALTERNATIVOS_LIDAR.items():
        for alias in aliases:
            if alias in df_padronizado.columns:
                mapeamento[alias] = nome_padrao
                break

    # Renomear colunas
    df_padronizado = df_padronizado.rename(columns=mapeamento)

    # Garantir que talhao e parcela sejam inteiros
    if 'talhao' in df_padronizado.columns:
        df_padronizado['talhao'] = pd.to_numeric(df_padronizado['talhao'], errors='coerce').astype('Int64')
    if 'parcela' in df_padronizado.columns:
        df_padronizado['parcela'] = pd.to_numeric(df_padronizado['parcela'], errors='coerce').astype('Int64')

    return df_padronizado


def limpar_dados_lidar(df_lidar):
    """
    Limpa e valida dados LiDAR

    Args:
        df_lidar: DataFrame com dados LiDAR

    Returns:
        DataFrame limpo
    """
    df_limpo = df_lidar.copy()

    # Remover linhas sem identificação de talhao/parcela
    df_limpo = df_limpo.dropna(subset=['talhao', 'parcela'])

    # Converter métricas para numérico
    colunas_numericas = [col for col in df_limpo.columns
                         if col not in ['talhao', 'parcela'] and
                         col in NOMES_ALTERNATIVOS_LIDAR.keys()]

    for col in colunas_numericas:
        df_limpo[col] = pd.to_numeric(df_limpo[col], errors='coerce')

    # Aplicar filtros de qualidade
    limites = CONFIGURACOES_LIDAR['limites_validacao']

    # Filtrar alturas irreais se altura_media disponível
    if 'altura_media' in df_limpo.columns:
        mask_altura = (df_limpo['altura_media'] >= limites['altura_min']) & \
                      (df_limpo['altura_media'] <= limites['altura_max'])
        df_limpo = df_limpo[mask_altura]

    # Remover outliers extremos usando IQR
    for col in colunas_numericas:
        if col in df_limpo.columns:
            Q1 = df_limpo[col].quantile(0.25)
            Q3 = df_limpo[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 3*IQR para ser menos restritivo
            upper_bound = Q3 + 3 * IQR

            df_limpo = df_limpo[(df_limpo[col] >= lower_bound) & (df_limpo[col] <= upper_bound)]

    return df_limpo


def calcular_metricas_derivadas_lidar(df_lidar):
    """
    Calcula métricas derivadas a partir das métricas básicas LiDAR

    Args:
        df_lidar: DataFrame com métricas básicas

    Returns:
        DataFrame com métricas derivadas adicionais
    """
    df_derivado = df_lidar.copy()

    # Índice de heterogeneidade vertical (se altura_media e desvio_altura disponíveis)
    if 'altura_media' in df_derivado.columns and 'desvio_altura' in df_derivado.columns:
        df_derivado['heterogeneidade_vertical'] = df_derivado['desvio_altura'] / df_derivado['altura_media']
        df_derivado['heterogeneidade_vertical'] = df_derivado['heterogeneidade_vertical'].fillna(0)

    # Índice de ocupação vertical (se cobertura e densidade disponíveis)
    if 'cobertura' in df_derivado.columns and 'densidade' in df_derivado.columns:
        df_derivado['ocupacao_vertical'] = df_derivado['densidade'] / df_derivado['cobertura']
        df_derivado['ocupacao_vertical'] = df_derivado['ocupacao_vertical'].fillna(0)

    # Razão altura máxima/média (indicador de dominância)
    if 'altura_maxima' in df_derivado.columns and 'altura_media' in df_derivado.columns:
        df_derivado['razao_max_media'] = df_derivado['altura_maxima'] / df_derivado['altura_media']
        df_derivado['razao_max_media'] = df_derivado['razao_max_media'].fillna(1)

    # Classificar complexidade estrutural
    if 'complexidade' in df_derivado.columns:
        df_derivado['classe_complexidade'] = pd.cut(
            df_derivado['complexidade'],
            bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
            labels=['Baixa', 'Média', 'Alta', 'Muito Alta']
        )

    return df_derivado


def integrar_dados_lidar_inventario(df_inventario, df_lidar):
    """
    Integra dados LiDAR com dados do inventário florestal

    Args:
        df_inventario: DataFrame do inventário
        df_lidar: DataFrame com métricas LiDAR

    Returns:
        DataFrame integrado
    """
    try:
        # Fazer merge por talhao e parcela
        df_integrado = df_inventario.merge(
            df_lidar,
            on=['talhao', 'parcela'],
            how='left',
            suffixes=('', '_lidar')
        )

        # Verificar cobertura da integração
        parcelas_com_lidar = df_integrado['altura_media'].notna().sum()
        parcelas_total = len(df_integrado)
        cobertura_percentual = (parcelas_com_lidar / parcelas_total) * 100

        st.info(f"📊 Integração LiDAR: {parcelas_com_lidar}/{parcelas_total} parcelas ({cobertura_percentual:.1f}%)")

        if cobertura_percentual < 50:
            st.warning("⚠️ Baixa cobertura LiDAR - verifique compatibilidade dos dados")

        return df_integrado

    except Exception as e:
        st.error(f"❌ Erro na integração: {e}")
        return df_inventario


def comparar_alturas_campo_lidar(df_integrado):
    """
    Compara alturas medidas em campo com alturas LiDAR

    Args:
        df_integrado: DataFrame com dados integrados

    Returns:
        dict: Estatísticas da comparação
    """
    # Filtrar apenas registros com ambas as medições
    mask_validas = df_integrado['H_m'].notna() & df_integrado['altura_media'].notna()
    df_comparacao = df_integrado[mask_validas].copy()

    if len(df_comparacao) == 0:
        st.warning("⚠️ Nenhuma parcela com dados de altura de campo e LiDAR")
        return None

    # Calcular estatísticas
    altura_campo = df_comparacao['H_m']
    altura_lidar = df_comparacao['altura_media']

    # Diferenças
    diferenca = altura_campo - altura_lidar
    diferenca_abs = np.abs(diferenca)
    diferenca_percentual = (diferenca / altura_lidar) * 100

    # Correlação
    correlacao = altura_campo.corr(altura_lidar)

    # Regressão linear
    X = altura_lidar.values.reshape(-1, 1)
    y = altura_campo.values

    modelo_reg = LinearRegression()
    modelo_reg.fit(X, y)

    y_pred = modelo_reg.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Estatísticas da comparação
    stats_comparacao = {
        'n_parcelas': len(df_comparacao),
        'correlacao': correlacao,
        'r2': r2,
        'rmse': rmse,
        'diferenca_media': diferenca.mean(),
        'diferenca_std': diferenca.std(),
        'diferenca_abs_media': diferenca_abs.mean(),
        'diferenca_percentual_media': diferenca_percentual.mean(),
        'coeficiente_angular': modelo_reg.coef_[0],
        'intercepto': modelo_reg.intercept_,
        'dados_comparacao': df_comparacao,
        'outliers': detectar_outliers_comparacao(df_comparacao)
    }

    return stats_comparacao


def detectar_outliers_comparacao(df_comparacao):
    """
    Detecta outliers na comparação campo vs LiDAR

    Args:
        df_comparacao: DataFrame com dados para comparação

    Returns:
        DataFrame com outliers identificados
    """
    diferenca_abs = np.abs(df_comparacao['H_m'] - df_comparacao['altura_media'])
    threshold = THRESHOLDS_ALERTAS_LIDAR['diferenca_altura_critica']

    outliers = df_comparacao[diferenca_abs > threshold].copy()
    outliers['diferenca_absoluta'] = diferenca_abs[diferenca_abs > threshold]

    return outliers


def calibrar_modelo_hipsometrico_com_lidar(df_integrado, modelo_original):
    """
    Calibra modelo hipsométrico usando dados LiDAR como referência

    Args:
        df_integrado: DataFrame com dados integrados
        modelo_original: Modelo hipsométrico original

    Returns:
        dict: Modelo calibrado e estatísticas
    """
    try:
        # Filtrar dados válidos
        mask_validas = (df_integrado['D_cm'].notna() &
                        df_integrado['altura_media'].notna() &
                        df_integrado['D_cm'] > 0)

        df_calibracao = df_integrado[mask_validas].copy()

        if len(df_calibracao) < 10:
            st.warning("⚠️ Poucos dados para calibração com LiDAR")
            return None

        # Preparar dados
        X = df_calibracao[['D_cm']].copy()
        y_lidar = df_calibracao['altura_media'].values

        # Modelo calibrado simples (linear)
        modelo_calibrado = LinearRegression()
        modelo_calibrado.fit(X, y_lidar)

        # Predições
        y_pred_calibrado = modelo_calibrado.predict(X)

        # Validação cruzada
        if CONFIGURACOES_LIDAR['parametros_calibracao']['validacao_cruzada']:
            kf = KFold(n_splits=CONFIGURACOES_LIDAR['parametros_calibracao']['k_folds'],
                       shuffle=True, random_state=42)
            cv_scores = cross_val_score(modelo_calibrado, X, y_lidar, cv=kf, scoring='r2')
            cv_score_medio = cv_scores.mean()
        else:
            cv_score_medio = None

        # Estatísticas
        r2_calibrado = r2_score(y_lidar, y_pred_calibrado)
        rmse_calibrado = np.sqrt(mean_squared_error(y_lidar, y_pred_calibrado))

        resultado_calibracao = {
            'modelo_calibrado': modelo_calibrado,
            'r2_calibrado': r2_calibrado,
            'rmse_calibrado': rmse_calibrado,
            'cv_score': cv_score_medio,
            'n_dados': len(df_calibracao),
            'coeficientes': {
                'intercepto': modelo_calibrado.intercept_,
                'slope': modelo_calibrado.coef_[0]
            },
            'dados_calibracao': df_calibracao,
            'predicoes_calibradas': y_pred_calibrado
        }

        return resultado_calibracao

    except Exception as e:
        st.error(f"❌ Erro na calibração: {e}")
        return None


def analisar_estrutura_florestal_lidar(df_lidar):
    """
    Análise da estrutura florestal baseada em métricas LiDAR

    Args:
        df_lidar: DataFrame com métricas LiDAR

    Returns:
        dict: Análise estrutural
    """
    analise = {}

    # Estatísticas básicas de altura
    if 'altura_media' in df_lidar.columns:
        analise['altura'] = {
            'media_geral': df_lidar['altura_media'].mean(),
            'std_geral': df_lidar['altura_media'].std(),
            'min': df_lidar['altura_media'].min(),
            'max': df_lidar['altura_media'].max(),
            'cv': (df_lidar['altura_media'].std() / df_lidar['altura_media'].mean()) * 100
        }

    # Análise de cobertura
    if 'cobertura' in df_lidar.columns:
        cobertura_media = df_lidar['cobertura'].mean()
        analise['cobertura'] = {
            'media': cobertura_media,
            'classe': classificar_cobertura(cobertura_media)
        }

    # Análise de complexidade
    if 'complexidade' in df_lidar.columns:
        analise['complexidade'] = {
            'media': df_lidar['complexidade'].mean(),
            'distribuicao': df_lidar['classe_complexidade'].value_counts().to_dict()
        }

    # Análise por talhão
    if 'talhao' in df_lidar.columns and 'altura_media' in df_lidar.columns:
        analise_talhao = df_lidar.groupby('talhao').agg({
            'altura_media': ['mean', 'std', 'count'],
            'cobertura': 'mean' if 'cobertura' in df_lidar.columns else lambda x: None
        })

        analise['por_talhao'] = analise_talhao.to_dict()

    return analise


def classificar_cobertura(cobertura_percentual):
    """
    Classifica o nível de cobertura florestal

    Args:
        cobertura_percentual: Percentual de cobertura

    Returns:
        str: Classificação da cobertura
    """
    if cobertura_percentual >= 90:
        return "Muito Alta"
    elif cobertura_percentual >= 75:
        return "Alta"
    elif cobertura_percentual >= 60:
        return "Média"
    elif cobertura_percentual >= 40:
        return "Baixa"
    else:
        return "Muito Baixa"


def gerar_alertas_automaticos_lidar(df_integrado, stats_comparacao):
    """
    Gera alertas automáticos baseados na análise LiDAR

    Args:
        df_integrado: DataFrame integrado
        stats_comparacao: Estatísticas da comparação

    Returns:
        list: Lista de alertas
    """
    alertas = []

    if stats_comparacao is None:
        alertas.append("⚠️ Não foi possível comparar dados campo vs LiDAR")
        return alertas

    # Alerta de correlação baixa
    if stats_comparacao['correlacao'] < THRESHOLDS_ALERTAS_LIDAR['r2_baixo']:
        alertas.append(f"🔴 Correlação baixa campo-LiDAR: {stats_comparacao['correlacao']:.3f}")

    # Alerta de outliers
    n_outliers = len(stats_comparacao['outliers'])
    if n_outliers > 0:
        pct_outliers = (n_outliers / stats_comparacao['n_parcelas']) * 100
        alertas.append(f"⚠️ {n_outliers} outliers detectados ({pct_outliers:.1f}%)")

    # Alerta de bias sistemático
    if abs(stats_comparacao['diferenca_media']) > 2.0:
        tendencia = "subestima" if stats_comparacao['diferenca_media'] > 0 else "superestima"
        alertas.append(f"📊 Bias sistemático: Campo {tendencia} em {abs(stats_comparacao['diferenca_media']):.1f}m")

    # Alerta de alta variabilidade
    if 'altura_media' in df_integrado.columns:
        cv_lidar = (df_integrado['altura_media'].std() / df_integrado['altura_media'].mean()) * 100
        if cv_lidar > THRESHOLDS_ALERTAS_LIDAR['cv_alto']:
            alertas.append(f"📈 Alta variabilidade LiDAR: CV = {cv_lidar:.1f}%")

    # Alerta de cobertura baixa
    if 'cobertura' in df_integrado.columns:
        cobertura_media = df_integrado['cobertura'].mean()
        if cobertura_media < THRESHOLDS_ALERTAS_LIDAR['cobertura_baixa']:
            alertas.append(f"🌲 Cobertura baixa detectada: {cobertura_media:.1f}%")

    return alertas


def calcular_metricas_validacao_lidar(df_integrado):
    """
    Calcula métricas de validação para dados LiDAR

    Args:
        df_integrado: DataFrame integrado

    Returns:
        dict: Métricas de validação
    """
    metricas = {}

    # Cobertura de dados LiDAR
    total_parcelas = len(df_integrado)
    parcelas_com_lidar = df_integrado['altura_media'].notna().sum()
    metricas['cobertura_lidar'] = (parcelas_com_lidar / total_parcelas) * 100

    # Qualidade das métricas
    if 'altura_media' in df_integrado.columns:
        dados_altura = df_integrado['altura_media'].dropna()
        metricas['qualidade_altura'] = {
            'n_dados': len(dados_altura),
            'valores_validos': (dados_altura > 0).sum(),
            'media': dados_altura.mean(),
            'std': dados_altura.std(),
            'outliers': len(detectar_outliers_iqr(dados_altura))
        }

    # Consistência entre talhões
    if 'talhao' in df_integrado.columns and 'altura_media' in df_integrado.columns:
        cv_por_talhao = df_integrado.groupby('talhao')['altura_media'].apply(
            lambda x: (x.std() / x.mean()) * 100 if len(x) > 1 else 0
        )
        metricas['consistencia_talhoes'] = {
            'cv_medio': cv_por_talhao.mean(),
            'talhoes_alta_variabilidade': (cv_por_talhao > 30).sum()
        }

    return metricas


def detectar_outliers_iqr(serie):
    """
    Detecta outliers usando método IQR

    Args:
        serie: Série pandas com dados

    Returns:
        Series: Índices dos outliers
    """
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = serie[(serie < lower_bound) | (serie > upper_bound)]
    return outliers


def exportar_dados_integrados(df_integrado):
    """
    Prepara dados integrados para exportação

    Args:
        df_integrado: DataFrame integrado

    Returns:
        str: CSV formatado para download
    """
    # Selecionar colunas principais
    colunas_export = ['talhao', 'parcela', 'D_cm', 'H_m']

    # Adicionar colunas LiDAR disponíveis
    colunas_lidar = ['altura_media', 'altura_maxima', 'desvio_altura',
                     'cobertura', 'densidade', 'complexidade']

    for col in colunas_lidar:
        if col in df_integrado.columns:
            colunas_export.append(col)

    # Adicionar métricas derivadas se disponíveis
    colunas_derivadas = ['heterogeneidade_vertical', 'ocupacao_vertical',
                         'razao_max_media', 'classe_complexidade']

    for col in colunas_derivadas:
        if col in df_integrado.columns:
            colunas_export.append(col)

    # Filtrar apenas colunas existentes
    colunas_existentes = [col for col in colunas_export if col in df_integrado.columns]

    df_export = df_integrado[colunas_existentes].copy()

    # Adicionar metadados
    df_export['fonte_altura'] = df_export.apply(
        lambda row: 'LiDAR' if pd.notna(row.get('altura_media')) else 'Campo',
        axis=1
    )

    df_export['data_processamento'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    return df_export.to_csv(index=False)


def gerar_relatorio_integracao_lidar(df_integrado, stats_comparacao, alertas):
    """
    Gera relatório completo da integração LiDAR

    Args:
        df_integrado: DataFrame integrado
        stats_comparacao: Estatísticas da comparação
        alertas: Lista de alertas

    Returns:
        str: Relatório em markdown
    """
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    relatorio = f"""# RELATÓRIO DE INTEGRAÇÃO LIDAR
**Data/Hora:** {timestamp}

## 📊 RESUMO GERAL
- **Total de parcelas:** {len(df_integrado)}
- **Parcelas com dados LiDAR:** {df_integrado['altura_media'].notna().sum() if 'altura_media' in df_integrado.columns else 0}
- **Cobertura LiDAR:** {(df_integrado['altura_media'].notna().sum() / len(df_integrado)) * 100:.1f}%

## 🔍 COMPARAÇÃO CAMPO vs LIDAR
"""

    if stats_comparacao:
        relatorio += f"""
- **Correlação:** {stats_comparacao['correlacao']:.3f}
- **R²:** {stats_comparacao['r2']:.3f}
- **RMSE:** {stats_comparacao['rmse']:.2f} m
- **Diferença média:** {stats_comparacao['diferenca_media']:.2f} m
- **Diferença absoluta média:** {stats_comparacao['diferenca_abs_media']:.2f} m
- **Outliers detectados:** {len(stats_comparacao['outliers'])}

### Equação de Calibração
**H_campo = {stats_comparacao['intercepto']:.3f} + {stats_comparacao['coeficiente_angular']:.3f} × H_lidar**
"""
    else:
        relatorio += "\n*Comparação não disponível - dados insuficientes*\n"

    # Alertas
    if alertas:
        relatorio += "\n## ⚠️ ALERTAS AUTOMÁTICOS\n"
        for alerta in alertas:
            relatorio += f"- {alerta}\n"

    # Métricas estruturais
    if 'altura_media' in df_integrado.columns:
        dados_altura = df_integrado['altura_media'].dropna()
        relatorio += f"""
## 🌲 ANÁLISE ESTRUTURAL
- **Altura média geral:** {dados_altura.mean():.1f} m
- **Desvio padrão:** {dados_altura.std():.1f} m
- **Altura mínima:** {dados_altura.min():.1f} m
- **Altura máxima:** {dados_altura.max():.1f} m
- **Coeficiente de variação:** {(dados_altura.std() / dados_altura.mean()) * 100:.1f}%
"""

    # Análise por talhão
    if 'talhao' in df_integrado.columns and 'altura_media' in df_integrado.columns:
        analise_talhao = df_integrado.groupby('talhao')['altura_media'].agg(['mean', 'std', 'count']).round(2)
        relatorio += "\n## 📊 ANÁLISE POR TALHÃO\n"
        for talhao, dados in analise_talhao.iterrows():
            relatorio += f"- **Talhão {talhao}:** {dados['mean']:.1f}m (±{dados['std']:.1f}m, n={dados['count']})\n"

    relatorio += f"""
## 🔧 CONFIGURAÇÕES UTILIZADAS
- **Altura mínima:** {CONFIGURACOES_LIDAR['limites_validacao']['altura_min']} m
- **Altura máxima:** {CONFIGURACOES_LIDAR['limites_validacao']['altura_max']} m
- **Threshold outliers:** {THRESHOLDS_ALERTAS_LIDAR['diferenca_altura_critica']} m
- **R² mínimo:** {CONFIGURACOES_LIDAR['limites_validacao']['r2_minimo']}

---
*Relatório gerado pelo Sistema Integrado de Inventário Florestal com dados LiDAR*
"""

    return relatorio