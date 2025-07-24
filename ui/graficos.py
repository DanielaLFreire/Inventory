# ui/graficos.py - VERSÃO CORRIGIDA
"""
Interface para gráficos e visualizações - COM CORREÇÃO DE MEMÓRIA
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils.formatacao import formatar_brasileiro, classificar_qualidade_modelo
from config.config import CORES_MODELOS
import contextlib


@contextlib.contextmanager
def criar_figura(*args, **kwargs):
    """Context manager para criar e limpar figuras automaticamente"""
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.close(fig)  # CORREÇÃO: Sempre fechar a figura


def criar_graficos_modelos(df_dados, resultados, predicoes, tipo_modelo):
    """
    Cria gráficos detalhados para modelos hipsométricos ou volumétricos

    Args:
        df_dados: DataFrame com dados originais
        resultados: Dict com resultados dos modelos
        predicoes: Dict com predições dos modelos
        tipo_modelo: 'hipsometrico' ou 'volumetrico'
    """
    if tipo_modelo == 'hipsometrico':
        criar_graficos_hipsometricos(df_dados, resultados, predicoes)
    elif tipo_modelo == 'volumetrico':
        criar_graficos_volumetricos(df_dados, resultados, predicoes)


def criar_graficos_hipsometricos(df_hip, resultados, predicoes):
    """
    Cria gráficos específicos para modelos hipsométricos

    Args:
        df_hip: DataFrame com dados hipsométricos
        resultados: Resultados dos modelos
        predicoes: Predições dos modelos
    """
    st.subheader("📊 Detalhamento dos Modelos Hipsométricos")

    # Criar abas para cada modelo
    if len(predicoes) > 0:
        abas_hip = st.tabs([f"{modelo}" for modelo in predicoes.keys()])

        for i, (modelo, aba) in enumerate(zip(predicoes.keys(), abas_hip)):
            with aba:
                mostrar_detalhes_modelo_hipsometrico(
                    modelo, df_hip, resultados[modelo], predicoes[modelo], i
                )


def mostrar_detalhes_modelo_hipsometrico(modelo, df_hip, resultado, predicao, cor_idx):
    """
    Mostra detalhes completos de um modelo hipsométrico

    Args:
        modelo: Nome do modelo
        df_hip: DataFrame com dados
        resultado: Resultado do modelo
        predicao: Predições do modelo
        cor_idx: Índice da cor
    """
    col1, col2 = st.columns([1, 1])

    with col1:
        # Informações do modelo
        mostrar_info_modelo_hipsometrico(modelo, resultado)

    with col2:
        # Gráfico principal do modelo
        criar_grafico_ajuste_hipsometrico(modelo, df_hip, predicao, cor_idx)

    # Gráficos de resíduos - com proteção extra
    try:
        criar_graficos_residuos_hipsometricos(modelo, df_hip['H_m'], predicao, cor_idx)
    except Exception as e:
        st.warning(f"⚠️ Gráficos de resíduos indisponíveis para {modelo}")
        st.info("💡 Continuando com outros modelos...")


def mostrar_info_modelo_hipsometrico(modelo, resultado):
    """
    Mostra informações de um modelo hipsométrico

    Args:
        modelo: Nome do modelo
        resultado: Resultado do modelo
    """
    r2g = resultado['r2g']
    rmse = resultado['rmse']

    # Classificação da qualidade
    qualidade = classificar_qualidade_modelo(r2g)

    st.write(f"**Qualidade:** {qualidade}")
    st.write(f"**R² Generalizado:** {r2g:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")

    # Equação específica do modelo
    mostrar_equacao_modelo_hipsometrico(modelo)


def mostrar_equacao_modelo_hipsometrico(modelo):
    """
    Mostra a equação específica do modelo hipsométrico

    Args:
        modelo: Nome do modelo
    """
    equacoes = {
        "Curtis": r"ln(H) = \beta_0 + \beta_1 \cdot \frac{1}{D}",
        "Campos": r"ln(H) = \beta_0 + \beta_1 \cdot \frac{1}{D} + \beta_2 \cdot ln(H_{dom})",
        "Henri": r"H = \beta_0 + \beta_1 \cdot ln(D)",
        "Prodan": r"\frac{D^2}{H-1.3} = \beta_0 + \beta_1 \cdot D + \beta_2 \cdot D^2",
        "Chapman": r"H = b_0 \cdot (1 - e^{-b_1 \cdot D})^{b_2}",
        "Weibull": r"H = a \cdot (1 - e^{-b \cdot D^c})",
        "Mononuclear": r"H = a \cdot (1 - b \cdot e^{-c \cdot D})"
    }

    if modelo in equacoes:
        st.latex(equacoes[modelo])


def criar_grafico_ajuste_hipsometrico(modelo, df_hip, predicao, cor_idx):
    """
    Cria gráfico de ajuste para modelo hipsométrico - COM CORREÇÃO DE MEMÓRIA

    Args:
        modelo: Nome do modelo
        df_hip: DataFrame com dados
        predicao: Predições do modelo
        cor_idx: Índice da cor
    """
    # CORREÇÃO: Usar context manager para gerenciar memória
    with criar_figura(figsize=(8, 6)) as (fig, ax):
        # Dados observados
        ax.scatter(df_hip['D_cm'], df_hip['H_m'], alpha=0.4, color='gray', s=15, label='Observado')

        # Modelo específico
        cor = CORES_MODELOS[cor_idx % len(CORES_MODELOS)]
        ax.scatter(df_hip['D_cm'], predicao, alpha=0.7, color=cor, s=15, label=f'{modelo}')

        # Configurações do gráfico
        r2_modelo = 1 - np.sum((df_hip['H_m'] - predicao) ** 2) / np.sum((df_hip['H_m'] - np.mean(df_hip['H_m'])) ** 2)
        ax.set_title(f'{modelo} (R² = {r2_modelo:.3f})')
        ax.set_xlabel('Diâmetro (cm)')
        ax.set_ylabel('Altura (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        # Figure será fechada automaticamente pelo context manager


def criar_graficos_residuos_hipsometricos(modelo, y_obs, y_pred, cor_idx):
    """
    Cria gráficos de resíduos para modelo hipsométrico - COM CORREÇÃO DE MEMÓRIA

    Args:
        modelo: Nome do modelo
        y_obs: Valores observados
        y_pred: Valores preditos
        cor_idx: Índice da cor
    """
    st.subheader(f"📈 Análise de Resíduos - {modelo}")

    try:
        # Converter tudo para numpy arrays primeiro
        y_obs_array = np.array(y_obs)
        y_pred_array = np.array(y_pred)

        # Garantir que ambos tenham o mesmo tamanho
        min_len = min(len(y_obs_array), len(y_pred_array))

        if min_len == 0:
            st.warning(f"⚠️ Sem dados válidos para resíduos do modelo {modelo}")
            return

        # Truncar para o mesmo tamanho
        y_obs_aligned = y_obs_array[:min_len]
        y_pred_aligned = y_pred_array[:min_len]

        # Verificar se ainda há incompatibilidade
        if len(y_obs_aligned) != len(y_pred_aligned):
            st.error(f"❌ Ainda há incompatibilidade de tamanhos: {len(y_obs_aligned)} vs {len(y_pred_aligned)}")
            return

        # Calcular resíduos
        residuos = y_obs_aligned - y_pred_aligned
        cor = CORES_MODELOS[cor_idx % len(CORES_MODELOS)]

        # Verificar se há dados NaN
        mask_validos = ~(np.isnan(y_pred_aligned) | np.isnan(residuos))
        y_pred_plot = y_pred_aligned[mask_validos]
        residuos_plot = residuos[mask_validos]

        if len(y_pred_plot) == 0:
            st.warning(f"⚠️ Sem dados válidos após remover NaN para {modelo}")
            return

        col1_res, col2_res = st.columns(2)

        with col1_res:
            # CORREÇÃO: Resíduos vs Preditos com context manager
            with criar_figura(figsize=(6, 5)) as (fig_res1, ax_res1):
                ax_res1.scatter(y_pred_plot, residuos_plot, alpha=0.6, color=cor)
                ax_res1.axhline(y=0, color='red', linestyle='--')
                ax_res1.set_xlabel('Valores Preditos (m)')
                ax_res1.set_ylabel('Resíduos (m)')
                ax_res1.set_title(f'Resíduos vs Preditos - {modelo}')
                ax_res1.grid(True, alpha=0.3)
                st.pyplot(fig_res1)
                # Figure será fechada automaticamente

        with col2_res:
            # CORREÇÃO: Histograma dos resíduos com context manager
            with criar_figura(figsize=(6, 5)) as (fig_res2, ax_res2):
                ax_res2.hist(residuos_plot, bins=min(15, len(residuos_plot) // 2), alpha=0.7, color=cor, edgecolor='black')
                ax_res2.axvline(x=0, color='red', linestyle='--')
                ax_res2.set_xlabel('Resíduos (m)')
                ax_res2.set_ylabel('Frequência')
                ax_res2.set_title(f'Distribuição dos Resíduos - {modelo}')
                ax_res2.grid(True, alpha=0.3)
                st.pyplot(fig_res2)
                # Figure será fechada automaticamente

    except Exception as e:
        # Criar gráfico básico sem resíduos como fallback
        st.info(f"💡 Criando gráfico simplificado para {modelo}")
        try:
            with criar_figura(figsize=(6, 4)) as (fig_simple, ax_simple):
                ax_simple.text(0.5, 0.5, f'Erro nos resíduos\ndo modelo {modelo}',
                               ha='center', va='center', transform=ax_simple.transAxes)
                ax_simple.set_title(f'Erro - {modelo}')
                st.pyplot(fig_simple)
                # Figure será fechada automaticamente
        except:
            pass


def criar_graficos_volumetricos(df_vol, resultados, predicoes):
    """
    Cria gráficos específicos para modelos volumétricos

    Args:
        df_vol: DataFrame com dados volumétricos
        resultados: Resultados dos modelos
        predicoes: Predições dos modelos
    """
    st.subheader("📊 Detalhamento dos Modelos Volumétricos")

    # Criar abas para cada modelo volumétrico
    abas_vol = st.tabs([f"{modelo}" for modelo in resultados.keys()])

    for i, (modelo, aba) in enumerate(zip(resultados.keys(), abas_vol)):
        with aba:
            mostrar_detalhes_modelo_volumetrico(
                modelo, df_vol, resultados[modelo], predicoes[modelo], i
            )


def mostrar_detalhes_modelo_volumetrico(modelo, df_vol, resultado, predicao, cor_idx):
    """
    Mostra detalhes completos de um modelo volumétrico

    Args:
        modelo: Nome do modelo
        df_vol: DataFrame com dados
        resultado: Resultado do modelo
        predicao: Predições do modelo
        cor_idx: Índice da cor
    """
    col1, col2 = st.columns([1, 1])

    with col1:
        # Informações do modelo
        mostrar_info_modelo_volumetrico(modelo, resultado)

    with col2:
        # Gráfico observado vs predito
        criar_grafico_obs_vs_pred_volumetrico(modelo, df_vol['V'], predicao, cor_idx)

    # Gráficos de resíduos volumétricos
    criar_graficos_residuos_volumetricos(modelo, df_vol['V'], predicao, cor_idx)


def mostrar_info_modelo_volumetrico(modelo, resultado):
    """
    Mostra informações de um modelo volumétrico

    Args:
        modelo: Nome do modelo
        resultado: Resultado do modelo
    """
    r2 = resultado['r2']
    rmse = resultado['rmse']

    # Classificação da qualidade
    qualidade = classificar_qualidade_modelo(r2)

    st.write(f"**Qualidade:** {qualidade}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")

    # Equação específica do modelo
    mostrar_equacao_modelo_volumetrico(modelo)


def mostrar_equacao_modelo_volumetrico(modelo):
    """
    Mostra a equação específica do modelo volumétrico

    Args:
        modelo: Nome do modelo
    """
    equacoes = {
        "Schumacher": r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot ln(H)",
        "G1": r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot \frac{1}{D}",
        "G2": r"V = \beta_0 + \beta_1 \cdot D^2 + \beta_2 \cdot D^2H + \beta_3 \cdot H",
        "G3": r"ln(V) = \beta_0 + \beta_1 \cdot ln(D^2H)"
    }

    if modelo in equacoes:
        st.latex(equacoes[modelo])


def criar_grafico_obs_vs_pred_volumetrico(modelo, y_obs, y_pred, cor_idx):
    """
    Cria gráfico observado vs predito para modelo volumétrico - COM CORREÇÃO DE MEMÓRIA

    Args:
        modelo: Nome do modelo
        y_obs: Valores observados
        y_pred: Valores preditos
        cor_idx: Índice da cor
    """
    # CORREÇÃO: Usar context manager
    with criar_figura(figsize=(8, 6)) as (fig, ax):
        # Scatter plot
        cor = CORES_MODELOS[cor_idx % len(CORES_MODELOS)]
        ax.scatter(y_obs, y_pred, alpha=0.6, color=cor)

        # Linha 1:1
        min_val = min(y_obs.min(), y_pred.min())
        max_val = max(y_obs.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1')

        # Configurações
        r2 = 1 - np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2)
        ax.set_title(f'{modelo} (R² = {r2:.3f})')
        ax.set_xlabel('Volume Observado (m³)')
        ax.set_ylabel('Volume Predito (m³)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        # Figure será fechada automaticamente


def criar_graficos_residuos_volumetricos(modelo, y_obs, y_pred, cor_idx):
    """
    Cria gráficos de resíduos para modelo volumétrico - COM CORREÇÃO DE MEMÓRIA

    Args:
        modelo: Nome do modelo
        y_obs: Valores observados
        y_pred: Valores preditos
        cor_idx: Índice da cor
    """
    st.subheader(f"📈 Análise de Resíduos - {modelo}")

    try:
        # Garantir que y_obs e y_pred tenham o mesmo tamanho
        if len(y_obs) != len(y_pred):
            # Alinhar os tamanhos (pegar o mínimo)
            min_len = min(len(y_obs), len(y_pred))
            y_obs_aligned = y_obs.iloc[:min_len] if hasattr(y_obs, 'iloc') else y_obs[:min_len]
            y_pred_aligned = y_pred[:min_len] if hasattr(y_pred, '__getitem__') else y_pred[:min_len]
        else:
            y_obs_aligned = y_obs
            y_pred_aligned = y_pred

        # Calcular resíduos com dados alinhados
        residuos = y_obs_aligned - y_pred_aligned
        cor = CORES_MODELOS[cor_idx % len(CORES_MODELOS)]

        # Converter para numpy arrays se necessário
        if hasattr(y_pred_aligned, 'values'):
            y_pred_plot = y_pred_aligned.values
        else:
            y_pred_plot = np.array(y_pred_aligned)

        if hasattr(residuos, 'values'):
            residuos_plot = residuos.values
        else:
            residuos_plot = np.array(residuos)

        col1_res, col2_res = st.columns(2)

        with col1_res:
            # CORREÇÃO: Resíduos vs Preditos com context manager
            with criar_figura(figsize=(6, 5)) as (fig_res1, ax_res1):
                ax_res1.scatter(y_pred_plot, residuos_plot, alpha=0.6, color=cor)
                ax_res1.axhline(y=0, color='red', linestyle='--')
                ax_res1.set_xlabel('Volumes Preditos (m³)')
                ax_res1.set_ylabel('Resíduos (m³)')
                ax_res1.set_title('Resíduos vs Preditos')
                ax_res1.grid(True, alpha=0.3)
                st.pyplot(fig_res1)
                # Figure será fechada automaticamente

        with col2_res:
            # CORREÇÃO: Histograma dos resíduos com context manager
            with criar_figura(figsize=(6, 5)) as (fig_res2, ax_res2):
                ax_res2.hist(residuos_plot, bins=15, alpha=0.7, color=cor, edgecolor='black')
                ax_res2.axvline(x=0, color='red', linestyle='--')
                ax_res2.set_xlabel('Resíduos (m³)')
                ax_res2.set_ylabel('Frequência')
                ax_res2.set_title('Distribuição dos Resíduos')
                ax_res2.grid(True, alpha=0.3)
                st.pyplot(fig_res2)
                # Figure será fechada automaticamente

    except Exception as e:
        st.error(f"❌ Erro ao criar gráficos de resíduos para {modelo}: {e}")
        st.info("💡 Possível incompatibilidade entre tamanhos dos dados observados e preditos")


def criar_graficos_inventario(resultados):
    """
    Cria gráficos específicos para o inventário final - COM CORREÇÃO DE MEMÓRIA

    Args:
        resultados: Resultados completos do inventário
    """
    resumo_parcelas = resultados.get('resumo_parcelas')
    resumo_talhoes = resultados.get('resumo_talhoes')

    if resumo_parcelas is None or len(resumo_parcelas) == 0:
        st.warning("⚠️ Dados insuficientes para gráficos")
        return

    # Verificar colunas disponíveis
    colunas_disponiveis = resumo_parcelas.columns.tolist()

    # Gráfico 1: Distribuição de produtividade (se vol_ha disponível)
    if 'vol_ha' in colunas_disponiveis:
        criar_grafico_distribuicao_produtividade(resumo_parcelas)
    else:
        st.info("ℹ️ Gráfico de produtividade não disponível (coluna vol_ha ausente)")

    # Gráfico 2: Produtividade por talhão (se dados de talhão disponíveis)
    if resumo_talhoes is not None and len(resumo_talhoes) > 0:
        criar_grafico_produtividade_talhao(resumo_talhoes)
    else:
        st.info("ℹ️ Gráfico por talhão não disponível")

    # Gráfico 3: Correlações (se colunas suficientes disponíveis)
    colunas_correlacao = ['vol_ha', 'dap_medio', 'altura_media', 'idade_anos']
    colunas_correlacao_disponiveis = [col for col in colunas_correlacao if col in colunas_disponiveis]

    if len(colunas_correlacao_disponiveis) >= 2:
        criar_graficos_correlacoes(resumo_parcelas)
    else:
        st.info("ℹ️ Gráficos de correlação não disponíveis (colunas insuficientes)")


def criar_grafico_distribuicao_produtividade(resumo_parcelas):
    """
    Cria gráfico de distribuição da produtividade - COM CORREÇÃO DE MEMÓRIA

    Args:
        resumo_parcelas: DataFrame com resumo por parcela
    """
    # Verificar se coluna vol_ha existe
    if 'vol_ha' not in resumo_parcelas.columns:
        st.warning("⚠️ Coluna 'vol_ha' não encontrada para gráfico de produtividade")
        return

    st.subheader("📊 Distribuição de Produtividade")

    col1, col2 = st.columns(2)

    with col1:
        try:
            # CORREÇÃO: Histograma com context manager
            with criar_figura(figsize=(8, 6)) as (fig, ax):
                vol_medio = resumo_parcelas['vol_ha'].mean()

                ax.hist(resumo_parcelas['vol_ha'], bins=15, alpha=0.7, color='forestgreen', edgecolor='black')
                ax.axvline(vol_medio, color='red', linestyle='--', linewidth=2,
                           label=f'Média: {vol_medio:.1f} m³/ha')
                ax.set_xlabel('Produtividade (m³/ha)')
                ax.set_ylabel('Frequência')
                ax.set_title('Distribuição de Produtividade')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                # Figure será fechada automaticamente
        except Exception as e:
            st.error(f"Erro no histograma: {e}")

    with col2:
        try:
            # CORREÇÃO: Box plot com context manager
            with criar_figura(figsize=(8, 6)) as (fig, ax):
                ax.boxplot(resumo_parcelas['vol_ha'], vert=True)
                ax.set_ylabel('Produtividade (m³/ha)')
                ax.set_title('Box Plot - Produtividade')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                # Figure será fechada automaticamente
        except Exception as e:
            st.error(f"Erro no box plot: {e}")


def criar_grafico_produtividade_talhao(resumo_talhoes):
    """
    Cria gráfico de produtividade por talhão - COM CORREÇÃO DE MEMÓRIA

    Args:
        resumo_talhoes: DataFrame com resumo por talhão
    """
    # Verificar se coluna de volume existe
    col_volume = None
    for col_name in ['vol_medio_ha', 'vol_ha', 'volume_medio']:
        if col_name in resumo_talhoes.columns:
            col_volume = col_name
            break

    if col_volume is None:
        st.warning("⚠️ Dados de volume por talhão não disponíveis")
        return

    st.subheader("🌳 Produtividade por Talhão")

    try:
        # CORREÇÃO: Gráfico de barras com context manager
        with criar_figura(figsize=(12, 6)) as (fig, ax):
            # Ordenar por produtividade
            talhao_ordenado = resumo_talhoes.sort_values(col_volume, ascending=False)

            bars = ax.bar(range(len(talhao_ordenado)),
                          talhao_ordenado[col_volume],
                          color='steelblue', alpha=0.7)

            ax.set_xlabel('Talhão')
            ax.set_ylabel('Produtividade (m³/ha)')
            ax.set_title('Produtividade por Talhão')
            ax.set_xticks(range(len(talhao_ordenado)))
            ax.set_xticklabels([f'T{t}' for t in talhao_ordenado['talhao']])
            ax.grid(True, alpha=0.3)

            # Adicionar valores nas barras
            for bar, val in zip(bars, talhao_ordenado[col_volume]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom')

            st.pyplot(fig)
            # Figure será fechada automaticamente

    except Exception as e:
        st.error(f"Erro no gráfico por talhão: {e}")


def criar_graficos_correlacoes(resumo_parcelas):
    """
    Cria gráficos de correlações entre variáveis - COM CORREÇÃO DE MEMÓRIA

    Args:
        resumo_parcelas: DataFrame com resumo por parcela
    """
    st.subheader("🔗 Correlações entre Variáveis")

    # Verificar quais colunas estão disponíveis
    colunas_interesse = {
        'vol_ha': 'Produtividade (m³/ha)',
        'dap_medio': 'DAP Médio (cm)',
        'altura_media': 'Altura Média (m)',
        'idade_anos': 'Idade (anos)'
    }

    colunas_disponiveis = {k: v for k, v in colunas_interesse.items() if k in resumo_parcelas.columns}

    if len(colunas_disponiveis) < 2:
        st.warning("⚠️ Colunas insuficientes para correlações")
        return

    try:
        # Determinar o layout baseado no número de gráficos possíveis
        n_graficos = min(4, len(colunas_disponiveis) * (len(colunas_disponiveis) - 1) // 2)

        # CORREÇÃO: Usar context manager para subplot
        if n_graficos >= 4:
            with criar_figura(figsize=(12, 10)) as (fig, axes):
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
                criar_subplots_correlacao(resumo_parcelas, colunas_disponiveis, axes)
                plt.tight_layout()
                st.pyplot(fig)
        elif n_graficos >= 2:
            with criar_figura(figsize=(12, 5)) as (fig, axes):
                if n_graficos == 2:
                    axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else [axes]
                else:
                    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
                criar_subplots_correlacao(resumo_parcelas, colunas_disponiveis, axes)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            with criar_figura(figsize=(6, 5)) as (fig, ax):
                axes = [ax]
                criar_subplots_correlacao(resumo_parcelas, colunas_disponiveis, axes)
                st.pyplot(fig)
        # Figure será fechada automaticamente pelo context manager

    except Exception as e:
        st.error(f"Erro nos gráficos de correlação: {e}")
        st.info(f"Colunas disponíveis: {list(colunas_disponiveis.keys())}")


def criar_subplots_correlacao(resumo_parcelas, colunas_disponiveis, axes):
    """
    Função auxiliar para criar subplots de correlação

    Args:
        resumo_parcelas: DataFrame com dados
        colunas_disponiveis: Dict com colunas disponíveis
        axes: Lista de axes do matplotlib
    """
    idx = 0
    cores = ['forestgreen', 'steelblue', 'orange', 'purple']

    # Gráfico 1: Produtividade vs DAP (se ambos disponíveis)
    if 'vol_ha' in colunas_disponiveis and 'dap_medio' in colunas_disponiveis and idx < len(axes):
        axes[idx].scatter(resumo_parcelas['dap_medio'], resumo_parcelas['vol_ha'],
                          alpha=0.6, color=cores[idx % len(cores)])
        axes[idx].set_xlabel(colunas_disponiveis['dap_medio'])
        axes[idx].set_ylabel(colunas_disponiveis['vol_ha'])
        axes[idx].set_title('Produtividade vs DAP')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Gráfico 2: Produtividade vs Altura (se ambos disponíveis)
    if 'vol_ha' in colunas_disponiveis and 'altura_media' in colunas_disponiveis and idx < len(axes):
        axes[idx].scatter(resumo_parcelas['altura_media'], resumo_parcelas['vol_ha'],
                          alpha=0.6, color=cores[idx % len(cores)])
        axes[idx].set_xlabel(colunas_disponiveis['altura_media'])
        axes[idx].set_ylabel(colunas_disponiveis['vol_ha'])
        axes[idx].set_title('Produtividade vs Altura')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Gráfico 3: Produtividade vs Idade (se ambos disponíveis)
    if 'vol_ha' in colunas_disponiveis and 'idade_anos' in colunas_disponiveis and idx < len(axes):
        axes[idx].scatter(resumo_parcelas['idade_anos'], resumo_parcelas['vol_ha'],
                          alpha=0.6, color=cores[idx % len(cores)])
        axes[idx].set_xlabel(colunas_disponiveis['idade_anos'])
        axes[idx].set_ylabel(colunas_disponiveis['vol_ha'])
        axes[idx].set_title('Produtividade vs Idade')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Gráfico 4: DAP vs Altura (se ambos disponíveis)
    if 'dap_medio' in colunas_disponiveis and 'altura_media' in colunas_disponiveis and idx < len(axes):
        axes[idx].scatter(resumo_parcelas['dap_medio'], resumo_parcelas['altura_media'],
                          alpha=0.6, color=cores[idx % len(cores)])
        axes[idx].set_xlabel(colunas_disponiveis['dap_medio'])
        axes[idx].set_ylabel(colunas_disponiveis['altura_media'])
        axes[idx].set_title('DAP vs Altura')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Esconder eixos não utilizados
    for i in range(idx, len(axes)):
        axes[i].set_visible(False)


def criar_grafico_classificacao_produtividade(stats):
    """
    Cria gráfico pizza da classificação de produtividade - COM CORREÇÃO DE MEMÓRIA

    Args:
        stats: Estatísticas gerais do inventário
    """
    # Verificar se dados estão disponíveis
    if not all(key in stats for key in ['classe_alta', 'classe_media', 'classe_baixa']):
        st.warning("⚠️ Dados de classificação não disponíveis")
        return

    st.subheader("🥧 Classificação de Produtividade")

    try:
        # Dados para o gráfico pizza
        labels = ['Classe Alta', 'Classe Média', 'Classe Baixa']
        sizes = [stats['classe_alta'], stats['classe_media'], stats['classe_baixa']]
        colors = ['#2e8b57', '#ffa500', '#dc143c']  # Verde, laranja, vermelho

        # CORREÇÃO: Usar context manager para gráfico pizza
        with criar_figura(figsize=(8, 8)) as (fig, ax):
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 12})

            ax.set_title('Distribuição das Classes de Produtividade', fontsize=14, fontweight='bold')

            # Adicionar legenda com detalhes
            q25 = stats.get('q25_volume', 80.0)
            q75 = stats.get('q75_volume', 120.0)

            legend_labels = [
                f'Alta (≥{q75:.1f} m³/ha): {stats["classe_alta"]} parcelas',
                f'Média ({q25:.1f}-{q75:.1f} m³/ha): {stats["classe_media"]} parcelas',
                f'Baixa (<{q25:.1f} m³/ha): {stats["classe_baixa"]} parcelas'
            ]

            ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            st.pyplot(fig)
            # Figure será fechada automaticamente

    except Exception as e:
        st.error(f"Erro no gráfico de classificação: {e}")


def configurar_estilo_matplotlib():
    """
    Configura estilo padrão para gráficos matplotlib
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.max_open_warning': 5  # CORREÇÃO: Reduzir limite de figuras abertas
    })


def limpar_cache_matplotlib():
    """
    NOVA FUNÇÃO: Limpa cache de matplotlib para liberar memória
    """
    try:
        plt.close('all')  # Fecha todas as figuras abertas
        plt.clf()         # Limpa figura atual
        plt.cla()         # Limpa axes atual
    except:
        pass


# CORREÇÃO: Configurar estilo ao importar o módulo
configurar_estilo_matplotlib()

# CORREÇÃO: Adicionar hook para limpeza automática
import atexit
atexit.register(limpar_cache_matplotlib)