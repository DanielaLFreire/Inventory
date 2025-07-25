# pages/2_📊_Modelos_Volumétricos.py - VERSÃO ADAPTADA PARA CONFIGURAÇÕES GLOBAIS
"""
Etapa 2: Modelos Volumétricos - USANDO CONFIGURAÇÕES CENTRALIZADAS
Cubagem e análise de modelos de volume com filtros globais
NOVO: Preparado para futuras extensões com parâmetros configuráveis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from processors.cubagem import processar_cubagem_smalian, calcular_estatisticas_cubagem
from models.volumetrico import ajustar_todos_modelos_volumetricos
from ui.graficos import criar_graficos_modelos
from utils.formatacao import formatar_brasileiro, classificar_qualidade_modelo

# NOVO: Importar configurações centralizadas
from config.configuracoes_globais import (
    obter_configuracao_global,
    aplicar_filtros_configuracao_global,
    mostrar_status_configuracao_sidebar
)

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
    """Verifica se pré-requisitos estão atendidos"""
    problemas = []

    # Verificar dados carregados
    if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
        problemas.append("Dados não carregados")

    # Verificar configuração global
    config_global = obter_configuracao_global()
    if not config_global.get('configurado', False):
        problemas.append("Sistema não configurado")

    if problemas:
        st.error("❌ Pré-requisitos não atendidos:")
        for problema in problemas:
            st.error(f"• {problema}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Página Principal", key="btn_principal_vol"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("⚙️ Configurações", key="btn_config_vol"):
                st.switch_page("pages/0_⚙️_Configurações.py")

        return False

    return True


def mostrar_configuracao_aplicada_cubagem():
    """Mostra configurações aplicadas especificamente para cubagem"""
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

    # NOVO: Mostrar configurações específicas para volumétricos (preparado para futuras extensões)
    mostrar_configuracoes_volumetricas_avancadas(config)

    # Botão para ajustar configurações
    if st.button("🔧 Ajustar Configurações", key="btn_ajustar_config_vol"):
        st.switch_page("pages/0_⚙️_Configurações.py")


def mostrar_configuracoes_volumetricas_avancadas(config):
    """NOVO: Mostra configurações avançadas para modelos volumétricos"""
    # NOTA: Atualmente modelos volumétricos são lineares, mas preparado para futuras extensões

    with st.expander("🔧 Configurações Avançadas dos Modelos Volumétricos"):
        st.info("💡 Configurações aplicadas nos modelos volumétricos")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**📊 Modelos Disponíveis:**")
            st.write("• Schumacher-Hall (logarítmico)")
            st.write("• G1 - Goulding (logarítmico)")
            st.write("• G2 - Linear direto")
            st.write("• G3 - Spurr (logarítmico)")

            # NOVO: Verificar se há configurações específicas para volumétricos
            vol_config = config.get('parametros_vol_nao_lineares', {})
            if vol_config.get('enabled', False):
                st.success("✅ Configurações não-lineares habilitadas")
            else:
                st.info("ℹ️ Usando apenas modelos lineares padrão")

        with col2:
            st.write("**⚙️ Parâmetros de Qualidade:**")
            st.write("• Validação automática de outliers")
            st.write("• Cálculo de R² tradicional")
            st.write("• Análise de resíduos")
            st.write("• Detecção de dados inconsistentes")

            # NOVO: Mostrar configurações de tolerância se disponíveis
            tolerancia = config.get('tolerancia_ajuste', 0.01)
            st.write(f"• Tolerância para validação: {tolerancia}")


def processar_cubagem_com_filtros():
    """Processa cubagem aplicando filtros das configurações globais"""
    st.header("🔄 Processamento da Cubagem")

    # NOVO: Aplicar filtros às árvores da cubagem baseado nas configurações globais
    config = obter_configuracao_global()
    df_cubagem_original = st.session_state.dados_cubagem

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

    # NOVO: Aplicar filtro de diâmetro mínimo na cubagem também
    diametro_min = config.get('diametro_min', 4.0)
    if 'D_cm' in df_cubagem_filtrada.columns:
        df_cubagem_filtrada = df_cubagem_filtrada[df_cubagem_filtrada['D_cm'] >= diametro_min]
        st.info(f"🔍 Filtro de diâmetro aplicado: >= {diametro_min} cm")

    # Verificar se há dados suficientes após filtros
    if len(df_cubagem_filtrada) < 10:
        st.error("❌ Poucos dados de cubagem após aplicar filtros")
        st.info("💡 Ajuste as configurações para incluir mais talhões")
        return None, None

    # Processar cubagem
    with st.spinner("Processando cubagem pelo método de Smalian..."):
        volumes_arvore = processar_cubagem_smalian(df_cubagem_filtrada)

    if len(volumes_arvore) < 5:
        st.error("❌ Poucos volumes válidos da cubagem após filtros")
        return None, None

    # NOVO: Aplicar validação adicional baseada nas configurações
    volumes_validados = aplicar_validacao_volumetrica(volumes_arvore, config)

    if len(volumes_validados) != len(volumes_arvore):
        st.info(f"🔍 Validação aplicada: {len(volumes_arvore)} → {len(volumes_validados)} volumes válidos")

    if len(volumes_validados) < 5:
        st.error("❌ Poucos volumes válidos após validação")
        return None, None

    # Calcular estatísticas
    stats_cubagem = calcular_estatisticas_cubagem(volumes_validados)

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
    mostrar_graficos_cubagem(volumes_validados, stats_cubagem)

    return volumes_validados, stats_cubagem


def aplicar_validacao_volumetrica(volumes_arvore, config):
    """NOVO: Aplica validação adicional baseada nas configurações"""
    df_validado = volumes_arvore.copy()

    # Remover outliers extremos baseado na tolerância configurada
    tolerancia = config.get('tolerancia_ajuste', 0.01)

    # Calcular limites baseados no percentil
    Q1 = df_validado['V'].quantile(0.25)
    Q3 = df_validado['V'].quantile(0.75)
    IQR = Q3 - Q1

    # Usar tolerância como fator multiplicador (mais restritivo = menos tolerância)
    fator_outlier = max(1.5, 3.0 * (1 - tolerancia * 10))  # Varia de 1.5 a 3.0

    limite_inferior = Q1 - fator_outlier * IQR
    limite_superior = Q3 + fator_outlier * IQR

    # Filtrar outliers
    mask_validos = (df_validado['V'] >= limite_inferior) & (df_validado['V'] <= limite_superior)
    df_validado = df_validado[mask_validos]

    # Validações adicionais
    df_validado = df_validado[
        (df_validado['V'] > 0.001) &  # Volume mínimo
        (df_validado['D_cm'] > 0) &  # DAP válido
        (df_validado['H_m'] > 1.3)  # Altura válida
        ]

    return df_validado


def mostrar_graficos_cubagem(volumes_arvore, stats_cubagem):
    """Mostra gráficos da análise de cubagem"""
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


def executar_analise_volumetrica(volumes_arvore):
    """Executa análise volumétrica com configurações centralizadas"""
    st.header("🚀 Executando Análise Volumétrica")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Preparando dados para modelagem...")
        progress_bar.progress(0.2)

        # NOVO: Obter configurações globais
        config = obter_configuracao_global()

        status_text.text("🧮 Ajustando modelos volumétricos...")
        progress_bar.progress(0.5)

        # NOVO: Passar configurações para a função (preparado para futuras extensões)
        resultados, predicoes, melhor_modelo = ajustar_todos_modelos_volumetricos_com_config(volumes_arvore, config)

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

        # NOVO: Mostrar informações sobre configurações aplicadas
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
            st.write(f"**Volumes disponíveis**: {len(volumes_arvore)} registros")
            st.write(f"**Configurações**: {config}")


def ajustar_todos_modelos_volumetricos_com_config(volumes_arvore, config):
    """NOVO: Wrapper que aplica configurações aos modelos volumétricos"""
    # Por enquanto, modelos volumétricos são lineares, mas preparado para extensões futuras

    # Aplicar validações baseadas na configuração
    tolerancia = config.get('tolerancia_ajuste', 0.01)

    # Usar função original mas com validação adicional
    resultados, predicoes, melhor_modelo = ajustar_todos_modelos_volumetricos(volumes_arvore)

    # NOVO: Aplicar filtros de qualidade baseados na configuração
    if resultados:
        # Filtrar modelos com R² muito baixo baseado na tolerância
        r2_minimo = max(0.5, 1.0 - tolerancia * 10)  # Varia de 0.5 a 0.9

        resultados_filtrados = {}
        predicoes_filtradas = {}

        for modelo, resultado in resultados.items():
            if resultado['r2'] >= r2_minimo:
                resultados_filtrados[modelo] = resultado
                predicoes_filtradas[modelo] = predicoes[modelo]
            else:
                st.warning(f"⚠️ Modelo {modelo} removido (R² = {resultado['r2']:.3f} < {r2_minimo:.3f})")

        if resultados_filtrados:
            # Redeterminar melhor modelo após filtros
            melhor_modelo_filtrado = max(resultados_filtrados.keys(), key=lambda k: resultados_filtrados[k]['r2'])
            return resultados_filtrados, predicoes_filtradas, melhor_modelo_filtrado
        else:
            st.warning("⚠️ Todos os modelos foram filtrados. Usando resultados originais.")

    return resultados, predicoes, melhor_modelo


def mostrar_info_configuracoes_volumetricas(config, resultados):
    """Mostra informações sobre como configurações foram aplicadas"""
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

            st.write(f"• Tolerância: {config.get('tolerancia_ajuste', 0.01)}")

        with col2:
            st.write("**📊 Resultados Obtidos:**")
            st.write(f"• Total de modelos: {len(resultados)}")
            st.write(f"• Método de cubagem: Smalian")
            st.write(f"• Validação: Automática")

            # NOVO: Mostrar informações sobre qualidade dos modelos
            if resultados:
                r2_medio = sum(r['r2'] for r in resultados.values()) / len(resultados)
                st.write(f"• R² médio: {r2_medio:.3f}")

        st.info("""
        💡 **Nota**: Os filtros são aplicados na fase de cubagem e validação, 
        garantindo que apenas dados válidos sejam usados na modelagem volumétrica.
        """)


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


def mostrar_ranking_volumetricos(resultados):
    """Mostra ranking dos modelos volumétricos"""
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


def mostrar_coeficientes_volumetricos(resultados):
    """Mostra coeficientes dos modelos volumétricos"""
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


def mostrar_downloads_volumetricos(resultados, predicoes, volumes_arvore, sufixo=""):
    """Mostra opções de download - VERSÃO COM KEYS ÚNICAS"""
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
        # NOVO: Relatório com configurações
        relatorio = gerar_relatorio_volumetrico_com_configuracoes(resultados, df_ranking)
        st.download_button(
            "📄 Relatório com Configurações",
            relatorio,
            "relatorio_volumetricos_configuracoes.md",
            "text/markdown",
            key=gerar_key_unica(f"download_relatorio_vol{sufixo}")
        )


def gerar_relatorio_volumetrico_com_configuracoes(resultados, df_ranking):
    """NOVO: Gera relatório incluindo configurações aplicadas"""
    config = obter_configuracao_global()
    melhor = df_ranking.iloc[0]

    relatorio = f"""# RELATÓRIO - MODELOS VOLUMÉTRICOS COM CONFIGURAÇÕES

## 🏆 MELHOR MODELO
**{melhor['Modelo']}** - {melhor['Qualidade']}
- R²: {melhor['R2']:.4f}
- RMSE: {melhor['RMSE']:.4f}

## ⚙️ CONFIGURAÇÕES APLICADAS
### Filtros na Cubagem:
- Diâmetro mínimo: {config.get('diametro_min', 4.0)} cm
- Talhões excluídos: {config.get('talhoes_excluir', [])}
- Tolerância de validação: {config.get('tolerancia_ajuste', 0.01)}

### Método de Cubagem:
- Método: Smalian
- Fórmula: V = (A₁ + A₂)/2 × L
- Precisão: ±2-5% do volume real
- Validação automática de outliers: ✅

### Configurações de Qualidade:
- R² mínimo aplicado: {max(0.5, 1.0 - config.get('tolerancia_ajuste', 0.01) * 10):.3f}
- Filtros de outliers automáticos: ✅
- Validação de dados inconsistentes: ✅

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
- Validação automática aplicada: ✅
- Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 RECOMENDAÇÃO
Use o modelo **{melhor['Modelo']}** para estimativas volumétricas neste povoamento.

## 🔧 CONFIGURAÇÕES TÉCNICAS
- Filtros aplicados na cubagem conforme configuração global
- Outliers removidos automaticamente baseado na tolerância configurada
- Modelos com baixo R² filtrados automaticamente
- Consistência garantida com outras etapas do sistema

---
*Relatório gerado pelo Sistema de Inventário Florestal com Configurações Centralizadas*
*Cubagem processada pelo método de Smalian com validação automática*
"""

    return relatorio


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
            • Tolerância configurável

            Isso garante consistência entre todas as etapas.
            """)


def main():
    # Verificar pré-requisitos
    if not verificar_prerequisitos():
        return

    st.title("📊 Modelos Volumétricos")
    st.markdown("### Cubagem e Análise de Volume com Configurações Centralizadas")

    # Mostrar status da configuração na sidebar
    mostrar_status_configuracao_sidebar()

    # Botão para limpar resultados anteriores (evita conflitos de keys)
    if st.button("🗑️ Limpar Resultados Anteriores", key="limpar_resultados_vol"):
        if 'resultados_volumetricos' in st.session_state:
            del st.session_state