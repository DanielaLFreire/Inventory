# pages/2_📊_Modelos_Volumétricos.py - VERSÃO CORRIGIDA
"""
Etapa 2: Modelos Volumétricos - USANDO CONFIGURAÇÕES CENTRALIZADAS
Cubagem e análise de modelos de volume com filtros globais
CORRIGIDO: Imports, verificações de pré-requisitos e tratamento de erros
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback

# Imports com tratamento de erro
try:
    from processors.cubagem import processar_cubagem_smalian, calcular_estatisticas_cubagem
    from models.volumetrico import ajustar_todos_modelos_volumetricos
    from ui.graficos import criar_graficos_modelos
    from utils.formatacao import formatar_brasileiro, classificar_qualidade_modelo
except ImportError as e:
    st.error(f"❌ Erro de importação: {e}")
    st.stop()

try:
    from config.configuracoes_globais import (
        obter_configuracao_global,
        aplicar_filtros_configuracao_global,
        mostrar_status_configuracao_sidebar
    )
except ImportError as e:
    st.error(f"❌ Erro ao importar configurações: {e}")
    st.stop()

st.set_page_config(
    page_title="Modelos Volumétricos",
    page_icon="📊",
    layout="wide"
)

def gerar_key_unica(base_key):
    """Gera uma key única para evitar conflitos"""
    timestamp = int(time.time() * 1000)
    return f"{base_key}_{timestamp}"


def verificar_prerequisitos():
    """Verifica se pré-requisitos estão atendidos - VERSÃO CORRIGIDA"""
    problemas = []

    # Verificar dados carregados
    if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
        problemas.append("Dados não carregados")

    # Verificar se dados existem no session_state
    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        problemas.append("Dados de inventário ausentes")

    if not hasattr(st.session_state, 'dados_cubagem') or st.session_state.dados_cubagem is None:
        problemas.append("Dados de cubagem ausentes")

    # Verificar configuração global
    try:
        config_global = obter_configuracao_global()
        if not config_global.get('configurado', False):
            problemas.append("Sistema não configurado")
    except Exception as e:
        st.warning(f"⚠️ Erro ao verificar configuração: {e}")
        problemas.append("Erro na configuração global")

    if problemas:
        st.error("❌ Pré-requisitos não atendidos:")
        for problema in problemas:
            st.error(f"• {problema}")

        # Botões de navegação
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏠 Página Principal", key="btn_principal_vol"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("⚙️ Configurações", key="btn_config_vol"):
                st.switch_page("pages/0_⚙️_Configurações.py")
        with col3:
            if st.button("🌳 Hipsométricos", key="btn_hip_vol"):
                st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")

        return False

    return True
def converter_dados_volumetricos_brasileiros(df_volumes):
    """
    Converte dados volumétricos do formato brasileiro usando validação existente

    Args:
        df_volumes: DataFrame com dados em formato brasileiro

    Returns:
        DataFrame com dados convertidos e validados
    """
    print("🇧🇷 Convertendo dados volumétricos do formato brasileiro...")

    df = df_volumes.copy()

    # Detectar e converter colunas numéricas
    colunas_converter = ['D_cm', 'H_m', 'V']

    for coluna in colunas_converter:
        if coluna in df.columns:
            print(f"  Processando {coluna}...")

            # Detectar tipo da coluna
            tipo_detectado = detectar_tipo_coluna(df[coluna], coluna)
            print(f"    Tipo detectado: {tipo_detectado}")

            # Converter valores do formato brasileiro
            def converter_valor_brasileiro(valor):
                if pd.isna(valor):
                    return np.nan
                if isinstance(valor, (int, float)):
                    return float(valor)
                if isinstance(valor, str):
                    valor = valor.strip()
                    if valor == '' or valor.lower() == 'nan':
                        return np.nan
                    try:
                        # Formato brasileiro: vírgula para decimal
                        valor_convertido = valor.replace(',', '.')
                        return float(valor_convertido)
                    except (ValueError, TypeError):
                        return np.nan
                return np.nan

            # Aplicar conversão
            valores_originais = df[coluna].iloc[:3].tolist()
            df[coluna] = df[coluna].apply(converter_valor_brasileiro)
            valores_convertidos = df[coluna].iloc[:3].tolist()

            print(f"    Exemplo conversão: {valores_originais} → {valores_convertidos}")

            # Validar usando função existente
            limites = {}
            if coluna == 'D_cm':
                limites = {'min': 1, 'max': 100}
            elif coluna == 'H_m':
                limites = {'min': 1, 'max': 50}
            elif coluna == 'V':
                limites = {'min': 0.001, 'max': 5}

            validacao = validar_dados_numericos(df[coluna], coluna, limites)

            if validacao['valida']:
                stats = validacao['estatisticas']
                print(f"    ✅ {stats['validos']}/{stats['total']} valores convertidos com sucesso")
            else:
                print(f"    ⚠️ Problemas na conversão:")
                for problema in validacao['problemas'][:2]:  # Mostrar só os primeiros 2
                    print(f"      • {problema}")

    return df


def mostrar_configuracao_aplicada_cubagem():
    """Mostra configurações aplicadas especificamente para cubagem"""
    try:
        config = obter_configuracao_global()

        with st.expander("⚙️ Configurações Aplicadas na Cubagem"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🔍 Filtros na Cubagem:**")
                st.write(f"• Diâmetro mínimo: {config.get('diametro_min', 4.0)} cm")

                talhoes_excluir = config.get('talhoes_excluir', [])
                if talhoes_excluir:
                    st.write(f"• Talhões excluídos: {talhoes_excluir}")
                    st.caption("   (Árvores destes talhões não serão usadas)")
                else:
                    st.write("• Talhões excluídos: Nenhum")

            with col2:
                st.write("**📏 Método de Cubagem:**")
                st.write("• Método: Smalian")
                st.write("• Fórmula: V = (A₁ + A₂)/2 × L")
                st.write("• Validação: Automática")
                st.caption("  Dados inconsistentes são removidos")

        # Botão para ajustar configurações
        if st.button("🔧 Ajustar Configurações", key="btn_ajustar_config_vol"):
            st.switch_page("pages/0_⚙️_Configurações.py")

    except Exception as e:
        st.warning(f"⚠️ Erro ao mostrar configuração: {e}")


def processar_cubagem_com_filtros():
    """Processa cubagem aplicando filtros das configurações globais - VERSÃO CORRIGIDA"""
    st.header("🔄 Processamento da Cubagem")

    try:
        # Aplicar filtros às árvores da cubagem baseado nos talhões excluídos
        config = obter_configuracao_global()
        df_cubagem_original = st.session_state.dados_cubagem

        # Verificar se dados de cubagem existem
        if df_cubagem_original is None or len(df_cubagem_original) == 0:
            st.error("❌ Dados de cubagem não disponíveis")
            return None, None

        # Filtrar talhões excluídos na cubagem
        talhoes_excluir = config.get('talhoes_excluir', [])
        if talhoes_excluir:
            df_cubagem_filtrada = df_cubagem_original[~df_cubagem_original['talhao'].isin(talhoes_excluir)]
            st.info(f"🔍 Filtros aplicados: Excluindo talhões {talhoes_excluir}")

            if len(df_cubagem_filtrada) != len(df_cubagem_original):
                st.write(f"• Registros originais: {len(df_cubagem_original)}")
                st.write(f"• Registros após filtro: {len(df_cubagem_filtrada)}")
        else:
            df_cubagem_filtrada = df_cubagem_original
            st.info("🔍 Nenhum filtro de talhão aplicado na cubagem")

        # Verificar se há dados suficientes após filtros
        if len(df_cubagem_filtrada) < 10:
            st.error("❌ Poucos dados de cubagem após aplicar filtros")
            st.info("💡 Ajuste as configurações para incluir mais talhões")
            return None, None

        # Processar cubagem
        with st.spinner("Processando cubagem pelo método de Smalian..."):
            volumes_arvore = processar_cubagem_smalian(df_cubagem_filtrada)

        if volumes_arvore is None or len(volumes_arvore) < 5:
            st.error("❌ Poucos volumes válidos da cubagem após filtros")
            return None, None

        # Aplicar filtro de diâmetro mínimo aos volumes calculados
        diametro_min = config.get('diametro_min', 4.0)
        volumes_filtrados = volumes_arvore[volumes_arvore['D_cm'] >= diametro_min]

        if len(volumes_filtrados) != len(volumes_arvore):
            st.info(f"🔍 Filtro de diâmetro aplicado: {len(volumes_arvore)} → {len(volumes_filtrados)} árvores")

        if len(volumes_filtrados) < 5:
            st.error("❌ Poucos volumes válidos após filtro de diâmetro")
            return None, None

        # Calcular estatísticas
        stats_cubagem = calcular_estatisticas_cubagem(volumes_filtrados)

        # Mostrar estatísticas
        st.subheader("📊 Estatísticas da Cubagem (Dados Filtrados)")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Árvores Cubadas", stats_cubagem['total_arvores'])
        with col2:
            st.metric("Volume Total", f"{stats_cubagem['volume_total']:.3f} m³")
        with col3:
            st.metric("Volume Médio", f"{stats_cubagem['volume_medio']:.4f} m³")
        with col4:
            st.metric("CV Volume", f"{stats_cubagem['cv_volume']:.1f}%")

        # Gráficos da cubagem
        mostrar_graficos_cubagem(volumes_filtrados, stats_cubagem)

        return volumes_filtrados, stats_cubagem

    except Exception as e:
        st.error(f"❌ Erro no processamento da cubagem: {e}")
        st.info("💡 Verifique os dados de cubagem e configurações")

        # Debug detalhado
        with st.expander("🔍 Debug Detalhado"):
            st.write(f"**Tipo do erro**: {type(e).__name__}")
            st.write(f"**Mensagem**: {str(e)}")
            st.code(traceback.format_exc())

        return None, None


def mostrar_graficos_cubagem(volumes_arvore, stats_cubagem):
    """Mostra gráficos da análise de cubagem - VERSÃO CORRIGIDA"""
    try:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(volumes_arvore['V'], bins=20, alpha=0.7, color='forestgreen', edgecolor='black')
            ax.axvline(stats_cubagem['volume_medio'], color='red', linestyle='--', linewidth=2,
                       label=f'Média: {stats_cubagem["volume_medio"]:.4f} m³')
            ax.set_xlabel('Volume (m³)')
            ax.set_ylabel('Frequência')
            ax.set_title('Distribuição dos Volumes das Árvores')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**Estatísticas Descritivas:**")
            st.write(f"• **Mínimo**: {stats_cubagem['volume_min']:.4f} m³")
            st.write(f"• **Máximo**: {stats_cubagem['volume_max']:.4f} m³")
            st.write(f"• **Mediana**: {volumes_arvore['V'].median():.4f} m³")
            st.write(f"• **Q1**: {volumes_arvore['V'].quantile(0.25):.4f} m³")
            st.write(f"• **Q3**: {volumes_arvore['V'].quantile(0.75):.4f} m³")
            st.write(f"• **Desvio Padrão**: {volumes_arvore['V'].std():.4f} m³")

            st.markdown("**Características Dendrométricas:**")
            st.write(f"• **DAP médio**: {stats_cubagem['dap_medio']:.1f} cm")
            st.write(f"• **Altura média**: {stats_cubagem['altura_media']:.1f} m")
            st.write(f"• **Coef. Variação**: {stats_cubagem['cv_volume']:.1f}%")

    except Exception as e:
        st.error(f"❌ Erro ao criar gráficos da cubagem: {e}")


def executar_analise_volumetrica(volumes_arvore):
    """Executa análise volumétrica com configurações centralizadas - VERSÃO CORRIGIDA"""
    st.header("🚀 Executando Análise Volumétrica")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Preparando dados para modelagem...")
        progress_bar.progress(0.2)

        # Obter configurações
        config = obter_configuracao_global()

        status_text.text("🧮 Ajustando modelos volumétricos...")
        progress_bar.progress(0.5)

        # Usar função modular existente
        resultados, predicoes, melhor_modelo = ajustar_todos_modelos_volumetricos(volumes_arvore)

        progress_bar.progress(1.0)
        status_text.text("✅ Análise concluída!")

        if not resultados:
            st.error("❌ Nenhum modelo volumétrico foi ajustado com sucesso")
            return

        # Salvar resultados com configuração aplicada
        st.session_state.resultados_volumetricos = {
            'resultados': resultados,
            'predicoes': predicoes,
            'melhor_modelo': melhor_modelo,
            'volumes': volumes_arvore,
            'config_aplicada': config,
            'timestamp': pd.Timestamp.now()
        }

        st.success(f"🏆 Melhor modelo: **{melhor_modelo}**")

        # Mostrar informações sobre configurações aplicadas
        mostrar_info_configuracoes_volumetricas(config, resultados)

        # Mostrar resultados
        mostrar_resultados_volumetricos(resultados, predicoes, volumes_arvore, contexto="novo")

    except Exception as e:
        st.error(f"❌ Erro na análise volumétrica: {e}")
        st.info("💡 Verifique os dados de cubagem e configurações")

        # Debug detalhado
        with st.expander("🔍 Debug Detalhado"):
            st.write(f"**Tipo do erro**: {type(e).__name__}")
            st.write(f"**Mensagem**: {str(e)}")
            st.write(f"**Volumes disponíveis**: {len(volumes_arvore) if volumes_arvore is not None else 0} registros")
            st.write(f"**Configurações**: {config}")
            st.code(traceback.format_exc())


def mostrar_info_configuracoes_volumetricas(config, resultados):
    """Mostra informações sobre como configurações foram aplicadas"""
    try:
        with st.expander("ℹ️ Como as Configurações Foram Aplicadas"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🔧 Configurações Usadas:**")
                st.write(f"• Diâmetro mínimo: {config.get('diametro_min', 4.0)} cm")
                talhoes_excluir = config.get('talhoes_excluir', [])
                if talhoes_excluir:
                    st.write(f"• Talhões excluídos: {talhoes_excluir}")
                else:
                    st.write("• Talhões excluídos: Nenhum")

            with col2:
                st.write("**📊 Resultados Obtidos:**")
                st.write(f"• Total de modelos: {len(resultados)}")
                st.write(f"• Método de cubagem: Smalian")
                st.write(f"• Validação: Automática")

            st.info("""
            💡 **Nota**: Os filtros são aplicados na fase de cubagem, 
            garantindo que apenas dados válidos sejam usados na modelagem.
            """)

    except Exception as e:
        st.warning(f"⚠️ Erro ao mostrar informações: {e}")


def mostrar_resultados_volumetricos(resultados, predicoes, volumes_arvore, contexto="novo"):
    """
    Mostra resultados dos modelos volumétricos - VERSÃO CORRIGIDA PARA KEYS

    Args:
        contexto: "novo" para execução atual, "salvo" para resultados salvos
    """
    st.header("📊 Resultados dos Modelos Volumétricos")

    # Adicionar identificador do contexto
    if contexto == "salvo":
        st.info("ℹ️ Exibindo resultados salvos da execução anterior")

    # Criar sufixo único baseado no contexto e timestamp
    sufixo = f"_{contexto}_{int(time.time())}"

    try:
        # Tabs para organizar resultados
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📊 Gráficos", "🔢 Coeficientes", "💾 Downloads"])

        with tab1:
            mostrar_ranking_volumetricos(resultados)

        with tab2:
            try:
                criar_graficos_modelos(volumes_arvore, resultados, predicoes, 'volumetrico')
            except Exception as e:
                st.error(f"Erro ao criar gráficos: {e}")
                st.info("Recarregue a página se o erro persistir")

        with tab3:
            mostrar_coeficientes_volumetricos(resultados)

        with tab4:
            mostrar_downloads_volumetricos(resultados, predicoes, volumes_arvore, sufixo)

    except Exception as e:
        st.error(f"❌ Erro ao mostrar resultados: {e}")


def mostrar_ranking_volumetricos(resultados):
    """Mostra ranking dos modelos volumétricos"""
    try:
        st.subheader("🏆 Ranking dos Modelos")

        ranking_data = []
        for modelo, resultado in resultados.items():
            r2 = resultado['r2']
            rmse = resultado['rmse']
            qualidade = classificar_qualidade_modelo(r2)

            ranking_data.append({
                'Ranking': 0,
                'Modelo': modelo,
                'R²': f"{r2:.4f}",
                'RMSE': f"{rmse:.4f}",
                'Qualidade': qualidade
            })

        # Ordenar por R²
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking = df_ranking.sort_values('R²', ascending=False)
        df_ranking['Ranking'] = range(1, len(df_ranking) + 1)

        st.dataframe(df_ranking, hide_index=True, use_container_width=True)

        # Destaque do melhor
        melhor = df_ranking.iloc[0]
        st.success(f"🏆 **Melhor modelo**: {melhor['Modelo']} (R² = {melhor['R²']})")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar ranking: {e}")


def mostrar_coeficientes_volumetricos(resultados):
    """Mostra coeficientes dos modelos volumétricos"""
    try:
        st.subheader("🔢 Coeficientes dos Modelos")

        for modelo, resultado in resultados.items():
            with st.expander(f"📊 {modelo} - Coeficientes"):
                modelo_obj = resultado.get('modelo')
                if modelo_obj and hasattr(modelo_obj, 'modelo'):
                    try:
                        coefs = modelo_obj.modelo.coef_
                        intercept = modelo_obj.modelo.intercept_

                        st.write(f"**Intercepto (β₀)**: {intercept:.6f}")

                        # Nomes específicos dos coeficientes por modelo
                        if modelo == 'Schumacher':
                            nomes_coef = ['β₁ (ln D)', 'β₂ (ln H)']
                        elif modelo == 'G1':
                            nomes_coef = ['β₁ (ln D)', 'β₂ (1/D)']
                        elif modelo == 'G2':
                            nomes_coef = ['β₁ (D²)', 'β₂ (D²H)', 'β₃ (H)']
                        elif modelo == 'G3':
                            nomes_coef = ['β₁ (ln D²H)']
                        else:
                            nomes_coef = [f'β{i + 1}' for i in range(len(coefs))]

                        for nome, coef in zip(nomes_coef, coefs):
                            st.write(f"**{nome}**: {coef:.6f}")

                    except Exception as e:
                        st.info(f"Coeficientes não disponíveis: {e}")

                # Estatísticas do modelo
                st.write(f"**R²**: {resultado['r2']:.4f}")
                st.write(f"**RMSE**: {resultado['rmse']:.4f}")

                # Interpretação específica
                if modelo == 'Schumacher':
                    st.info(
                        "💡 **Interpretação**: β₁ e β₂ representam elasticidades (% de mudança em V para 1% de mudança na variável)")

    except Exception as e:
        st.error(f"❌ Erro ao mostrar coeficientes: {e}")


def mostrar_downloads_volumetricos(resultados, predicoes, volumes_arvore, sufixo=""):
    """Mostra opções de download - VERSÃO COM KEYS ÚNICAS"""
    try:
        st.subheader("💾 Downloads")

        # Ranking dos modelos
        ranking_data = []
        for modelo, resultado in resultados.items():
            ranking_data.append({
                'Modelo': modelo,
                'R2': resultado['r2'],
                'RMSE': resultado['rmse'],
                'Qualidade': classificar_qualidade_modelo(resultado['r2'])
            })

        df_ranking = pd.DataFrame(ranking_data).sort_values('R2', ascending=False)

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_ranking = df_ranking.to_csv(index=False)
            st.download_button(
                "📊 Ranking dos Modelos",
                csv_ranking,
                "ranking_modelos_volumetricos.csv",
                "text/csv",
                key=gerar_key_unica(f"download_ranking_vol{sufixo}")
            )

        with col2:
            # Dados com predições
            df_resultado = volumes_arvore.copy()
            for modelo, pred in predicoes.items():
                df_resultado[f'pred_{modelo}'] = pred
                df_resultado[f'residuo_{modelo}'] = volumes_arvore['V'] - pred

            csv_dados = df_resultado.to_csv(index=False)
            st.download_button(
                "📄 Dados com Predições",
                csv_dados,
                "dados_predicoes_volumetricas.csv",
                "text/csv",
                key=gerar_key_unica(f"download_dados_vol{sufixo}")
            )

        with col3:
            # Relatório com configurações
            relatorio = gerar_relatorio_volumetrico_centralizado(resultados, df_ranking)
            st.download_button(
                "📄 Relatório Completo",
                relatorio,
                "relatorio_volumetricos_completo.md",
                "text/markdown",
                key=gerar_key_unica(f"download_relatorio_vol{sufixo}")
            )

    except Exception as e:
        st.error(f"❌ Erro nos downloads: {e}")


def gerar_relatorio_volumetrico_centralizado(resultados, df_ranking):
    """Gera relatório incluindo configurações aplicadas"""
    try:
        config = obter_configuracao_global()
        melhor = df_ranking.iloc[0]

        relatorio = f"""# RELATÓRIO - MODELOS VOLUMÉTRICOS

## 🏆 MELHOR MODELO
**{melhor['Modelo']}** - {melhor['Qualidade']}
- R²: {melhor['R2']:.4f}
- RMSE: {melhor['RMSE']:.4f}

## ⚙️ CONFIGURAÇÕES APLICADAS
### Filtros na Cubagem:
- Diâmetro mínimo: {config.get('diametro_min', 4.0)} cm
- Talhões excluídos: {config.get('talhoes_excluir', [])}

### Método de Cubagem:
- Método: Smalian
- Fórmula: V = (A₁ + A₂)/2 × L
- Precisão: ±2-5% do volume real

## 📊 RANKING COMPLETO
"""

        for i, (_, row) in enumerate(df_ranking.iterrows(), 1):
            relatorio += f"\n{i}. **{row['Modelo']}** - {row['Qualidade']}"
            relatorio += f"\n   - R²: {row['R2']:.4f}, RMSE: {row['RMSE']:.4f}\n"

        relatorio += f"""
## 📈 RESUMO DA ANÁLISE
- Total de modelos avaliados: {len(resultados)}
- Modelos logarítmicos: {len([m for m in resultados.keys() if m in ['Schumacher', 'G1', 'G3']])}
- Modelos lineares: {len([m for m in resultados.keys() if m in ['G2']])}
- Configuração centralizada aplicada: ✅
- Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 RECOMENDAÇÃO
Use o modelo **{melhor['Modelo']}** para estimativas volumétricas neste povoamento.

---
*Relatório gerado pelo Sistema de Inventário Florestal com Configurações Centralizadas*
"""

        return relatorio

    except Exception as e:
        return f"Erro ao gerar relatório: {e}"


def mostrar_fundamentos_smalian():
    """Mostra fundamentos do método de Smalian"""
    with st.expander("📏 Método de Smalian - Fundamentos Teóricos"):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("""
            **Método de Smalian** para cálculo do volume de seções:

            **Fórmula Fundamental:**
            """)
            st.latex(r"V = \frac{A_1 + A_2}{2} \times L")

            st.markdown("""
            **Onde:**
            - **V** = Volume da seção (m³)
            - **A₁** = Área da seção inferior (m²)
            - **A₂** = Área da seção superior (m²)
            - **L** = Comprimento da seção (m)

            **Área seccional:** A = π × (d/2)²

            **Princípio:** Assume que cada seção tem formato de tronco de cone,
            calculando o volume pela média das áreas das bases multiplicada pela altura.
            """)

            st.info("""
            💡 **Características do Método:**
            - Assume forma tronco-cônica entre seções
            - Precisão adequada para seções de 2m
            - Método padrão internacional (IUFRO)
            - Erro típico: ±2-5% comparado ao volume real
            """)

        with col2:
            st.info("""
            **🔍 Filtros Aplicados:**

            Os filtros das configurações globais são aplicados:

            • Talhões excluídos
            • Diâmetro mínimo
            • Validação automática

            Isso garante consistência entre todas as etapas.
            """)


def main():
    """Função principal da página - VERSÃO CORRIGIDA"""
    try:
        # Verificar pré-requisitos
        if not verificar_prerequisitos():
            return

        st.title("📊 Modelos Volumétricos")
        st.markdown("### Cubagem e Análise de Volume")

        # NOVO: Mostrar status da configuração na sidebar
        try:
            mostrar_status_configuracao_sidebar()
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erro na sidebar: {e}")

        # Botão para limpar resultados anteriores (evita conflitos de keys)
        if st.button("🗑️ Limpar Resultados Anteriores", key="limpar_resultados_vol"):
            if 'resultados_volumetricos' in st.session_state:
                del st.session_state.resultados_volumetricos
                st.success("✅ Resultados limpos!")
                st.rerun()

        # Mostrar configurações aplicadas
        mostrar_configuracao_aplicada_cubagem()

        # Fundamentos teóricos
        mostrar_fundamentos_smalian()

        # Informações dos modelos
        with st.expander("📚 Informações dos Modelos Volumétricos"):
            st.markdown("""
            ### 🧮 Modelos Disponíveis

            **Schumacher-Hall:**
            - ln(V) = β₀ + β₁×ln(D) + β₂×ln(H)
            - Modelo clássico, biologicamente realista

            **G1 (Goulding):**
            - ln(V) = β₀ + β₁×ln(D) + β₂×(1/D)
            - Substitui altura pelo inverso do diâmetro

            **G2 (Linear):**
            - V = β₀ + β₁×D² + β₂×D²H + β₃×H
            - Modelo linear, interpretação direta

            **G3 (Spurr):**
            - ln(V) = β₀ + β₁×ln(D²H)
            - Extremamente simples, uma variável
            """)

        # Processar cubagem com filtros
        resultado_cubagem = processar_cubagem_com_filtros()

        if resultado_cubagem[0] is None:
            return

        volumes_arvore, stats_cubagem = resultado_cubagem

        # Botão para executar análise
        if st.button("🚀 Executar Análise Volumétrica", type="primary", use_container_width=True):
            executar_analise_volumetrica(volumes_arvore)

        # Mostrar resultados salvos se existirem - COM CONTROLE PARA EVITAR KEYS DUPLICADAS
        if hasattr(st.session_state, 'resultados_volumetricos') and st.session_state.resultados_volumetricos:
            st.markdown("---")
            st.subheader("📂 Resultados Salvos")

            resultados_salvos = st.session_state.resultados_volumetricos

            # Verificar se configuração mudou
            try:
                config_atual = obter_configuracao_global()
                config_salva = resultados_salvos.get('config_aplicada', {})

                if config_atual != config_salva:
                    st.warning("""
                    ⚠️ **Configurações Alteradas**

                    As configurações globais foram modificadas desde a última execução.
                    Os resultados abaixo podem não refletir as configurações atuais.

                    **Recomendação**: Execute a análise novamente para aplicar as novas configurações.
                    """)
            except Exception as e:
                st.warning(f"⚠️ Erro ao comparar configurações: {e}")

            # Checkbox para controlar exibição e evitar conflitos
            if st.checkbox("👀 Mostrar Resultados Salvos", key="mostrar_resultados_salvos_vol"):
                mostrar_resultados_volumetricos(
                    resultados_salvos['resultados'],
                    resultados_salvos['predicoes'],
                    resultados_salvos['volumes'],
                    "salvo"  # CONTEXTO DIFERENTE PARA EVITAR CONFLITO DE KEYS
                )

    except Exception as e:
        st.error(f"❌ Erro crítico na página: {e}")
        st.info("💡 Tente recarregar a página")

        # Debug para ajudar na resolução
        with st.expander("🔍 Debug - Informações do Erro"):
            st.write(f"**Tipo do erro**: {type(e).__name__}")
            st.write(f"**Mensagem**: {str(e)}")
            st.code(traceback.format_exc())

        # Botões de navegação de emergência
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Voltar ao Início", key="btn_home_erro"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("⚙️ Configurações", key="btn_config_erro"):
                st.switch_page("pages/0_⚙️_Configurações.py")


if __name__ == "__main__":
    main()