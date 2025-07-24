# pages/2_📊_Modelos_Volumétricos.py
"""
Etapa 2: Modelos Volumétricos
Cubagem e análise de modelos de volume
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from processors.cubagem import processar_cubagem_smalian, calcular_estatisticas_cubagem
from models.volumetrico import ajustar_todos_modelos_volumetricos
from ui.graficos import criar_graficos_modelos
from utils.formatacao import formatar_brasileiro, classificar_qualidade_modelo

st.set_page_config(
    page_title="Modelos Volumétricos",
    page_icon="📊",
    layout="wide"
)


def verificar_dados():
    """Verifica se dados estão disponíveis"""
    if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
        st.error("❌ Dados não carregados. Volte à página principal.")
        if st.button("🏠 Voltar à Página Principal", key="btn_voltar_vol"):
            st.switch_page("app.py")
        return False
    return True


def obter_info_modelos_volumetricos():
    """Informações dos modelos volumétricos"""
    return {
        "Schumacher": {
            "equacao": r"\ln(V) = \beta_0 + \beta_1 \cdot \ln(D) + \beta_2 \cdot \ln(H)",
            "nome_completo": "Schumacher-Hall",
            "descricao": "Modelo clássico que relaciona logaritmo do volume com logaritmos das variáveis dendrométricas"
        },
        "G1": {
            "equacao": r"\ln(V) = \beta_0 + \beta_1 \cdot \ln(D) + \beta_2 \cdot \frac{1}{D}",
            "nome_completo": "Modelo G1 (Goulding)",
            "descricao": "Variação do Schumacher que substitui altura pelo inverso do diâmetro"
        },
        "G2": {
            "equacao": r"V = \beta_0 + \beta_1 \cdot D^2 + \beta_2 \cdot D^2H + \beta_3 \cdot H",
            "nome_completo": "Modelo G2 (Linear)",
            "descricao": "Modelo linear que combina área basal e altura de forma aditiva"
        },
        "G3": {
            "equacao": r"\ln(V) = \beta_0 + \beta_1 \cdot \ln(D^2H)",
            "nome_completo": "Modelo G3 (Spurr)",
            "descricao": "Modelo de Spurr usando produto D²H como única variável preditora"
        }
    }


def mostrar_fundamentos_smalian():
    """Mostra fundamentos do método de Smalian"""
    with st.expander("📏 Método de Smalian - Fundamentos Teóricos"):
        col1, col2 = st.columns([2, 1])

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
            # Gráfico ilustrativo
            fig, ax = plt.subplots(figsize=(6, 8))

            # Simular formato de tronco
            h = np.linspace(0, 20, 100)
            d = 25 * np.exp(-0.05 * h)

            ax.plot(d / 2, h, 'b-', linewidth=2, label='Perfil do tronco')
            ax.plot(-d / 2, h, 'b-', linewidth=2)

            # Mostrar seções
            secoes_h = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            for i, h_sec in enumerate(secoes_h):
                d_sec = 25 * np.exp(-0.05 * h_sec)
                ax.plot([-d_sec / 2, d_sec / 2], [h_sec, h_sec], 'r-', linewidth=3, alpha=0.7)
                if i % 2 == 0:
                    ax.text(d_sec / 2 + 1, h_sec, f'{h_sec}m', fontsize=8)

            ax.set_xlabel('Raio (cm)')
            ax.set_ylabel('Altura (m)')
            ax.set_title('Seções para Cubagem')
            ax.grid(True, alpha=0.3)
            ax.legend()

            st.pyplot(fig)
            plt.close()


def processar_cubagem():
    """Processa dados de cubagem"""
    st.header("🔄 Processamento da Cubagem")

    with st.spinner("Processando cubagem pelo método de Smalian..."):
        volumes_arvore = processar_cubagem_smalian(st.session_state.dados_cubagem)

    if len(volumes_arvore) < 10:
        st.error("❌ Poucos volumes válidos da cubagem. Verifique os dados.")
        return None

    # Estatísticas da cubagem
    stats_cubagem = calcular_estatisticas_cubagem(volumes_arvore)

    st.subheader("📊 Estatísticas da Cubagem")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Árvores Cubadas", stats_cubagem['total_arvores'])
    with col2:
        st.metric("Volume Total", f"{stats_cubagem['volume_total']:.3f} m³")
    with col3:
        st.metric("Volume Médio", f"{stats_cubagem['volume_medio']:.4f} m³")
    with col4:
        st.metric("CV Volume", f"{stats_cubagem['cv_volume']:.1f}%")

    # Distribuição dos volumes
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
        plt.close()

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

    return volumes_arvore, stats_cubagem


def mostrar_info_modelos():
    """Mostra informações dos modelos volumétricos"""
    with st.expander("📚 Informações dos Modelos Volumétricos"):
        modelos_info = obter_info_modelos_volumetricos()

        for modelo, info in modelos_info.items():
            with st.expander(f"📖 {modelo} - {info['nome_completo']}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Equação:**")
                    st.latex(info['equacao'])
                    st.markdown(f"**Descrição:** {info['descricao']}")

                with col2:
                    # Adicionar características específicas
                    if modelo == 'Schumacher':
                        st.markdown("**Vantagens:**")
                        st.markdown("• Biologicamente realista\n• Amplamente validado\n• Boa precisão geral")
                        st.markdown("**Limitações:**")
                        st.markdown("• Viés da transformação logarítmica\n• Requer correção para escala original")

                    elif modelo == 'G1':
                        st.markdown("**Vantagens:**")
                        st.markdown("• Requer apenas dados de DAP\n• Reduz erros de medição")
                        st.markdown("**Limitações:**")
                        st.markdown("• Menor precisão que modelos com altura")

                    elif modelo == 'G2':
                        st.markdown("**Vantagens:**")
                        st.markdown("• Interpretação direta\n• Sem correção de viés")
                        st.markdown("**Limitações:**")
                        st.markdown("• Pode gerar volumes negativos\n• Heterocedasticidade comum")

                    elif modelo == 'G3':
                        st.markdown("**Vantagens:**")
                        st.markdown("• Extremamente simples\n• Rápido de ajustar")
                        st.markdown("**Limitações:**")
                        st.markdown("• Menor precisão\n• Perda de informação")


def executar_analise_volumetrica(volumes_arvore):
    """Executa análise dos modelos volumétricos"""
    st.header("🚀 Executando Análise Volumétrica")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Preparando dados para modelagem...")
        progress_bar.progress(0.2)

        # Usar função modular existente
        resultados, predicoes, melhor_modelo = ajustar_todos_modelos_volumetricos(volumes_arvore)

        progress_bar.progress(1.0)
        status_text.text("✅ Análise concluída!")

        if not resultados:
            st.error("❌ Nenhum modelo volumétrico foi ajustado com sucesso")
            return

        # Salvar no session_state
        st.session_state.resultados_volumetricos = {
            'resultados': resultados,
            'predicoes': predicoes,
            'melhor_modelo': melhor_modelo,
            'volumes': volumes_arvore
        }

        st.success(f"🏆 Melhor modelo: **{melhor_modelo}**")

        # Mostrar resultados
        mostrar_resultados_volumetricos(resultados, predicoes, volumes_arvore, "execucao")

    except Exception as e:
        st.error(f"❌ Erro na análise volumétrica: {e}")
        st.info("💡 Verifique os dados de cubagem e tente novamente")


def mostrar_resultados_volumetricos(resultados, predicoes, volumes_arvore, contexto="atual"):
    """Mostra resultados dos modelos volumétricos"""
    st.header("📊 Resultados dos Modelos Volumétricos")

    # Tabs para organizar resultados
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📊 Gráficos", "🔢 Coeficientes", "💾 Downloads"])

    with tab1:
        # Ranking dos modelos
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

    with tab2:
        # Gráficos usando função modular
        criar_graficos_modelos(volumes_arvore, resultados, predicoes, 'volumetrico')

    with tab3:
        # Coeficientes detalhados
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

    with tab4:
        # Downloads com keys únicos baseados no contexto
        st.subheader("💾 Downloads")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Ranking dos modelos
            csv_ranking = df_ranking.to_csv(index=False)
            st.download_button(
                "📊 Ranking dos Modelos",
                csv_ranking,
                "ranking_modelos_volumetricos.csv",
                "text/csv",
                key=f"download_ranking_volumetricos_{contexto}"
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
                key=f"download_dados_volumetricos_{contexto}"
            )

        with col3:
            # Relatório resumido
            relatorio = gerar_relatorio_volumetrico(resultados, df_ranking)
            st.download_button(
                "📄 Relatório Resumido",
                relatorio,
                "relatorio_volumetricos.md",
                "text/markdown",
                key=f"download_relatorio_volumetricos_{contexto}"
            )


def gerar_relatorio_volumetrico(resultados, df_ranking):
    """Gera relatório dos modelos volumétricos"""
    melhor = df_ranking.iloc[0]

    relatorio = f"""# RELATÓRIO - MODELOS VOLUMÉTRICOS

## 🏆 MELHOR MODELO
**{melhor['Modelo']}** - {melhor['Qualidade']}
- R²: {melhor['R²']}
- RMSE: {melhor['RMSE']}

## 📊 RANKING COMPLETO
"""

    for _, row in df_ranking.iterrows():
        relatorio += f"\n{row['Ranking']}. **{row['Modelo']}** - {row['Qualidade']}"
        relatorio += f"\n   - R²: {row['R²']}, RMSE: {row['RMSE']}\n"

    relatorio += f"""
## 📏 MÉTODO DE CUBAGEM
- **Método**: Smalian
- **Fórmula**: V = (A₁ + A₂)/2 × L
- **Precisão**: ±2-5% do volume real

## 📈 RESUMO DA ANÁLISE
- Total de modelos avaliados: {len(resultados)}
- Modelos logarítmicos: {len([m for m in resultados.keys() if m in ['Schumacher', 'G1', 'G3']])}
- Modelos lineares: {len([m for m in resultados.keys() if m in ['G2']])}

## 🎯 RECOMENDAÇÃO
Use o modelo **{melhor['Modelo']}** para estimativas volumétricas neste povoamento.

---
*Relatório gerado pelo Sistema de Inventário Florestal*
"""

    return relatorio


def main():
    if not verificar_dados():
        return

    st.title("📊 Modelos Volumétricos")
    st.markdown("### Cubagem e Análise de Volume")

    # Fundamentos teóricos
    mostrar_fundamentos_smalian()

    # Informações dos modelos
    mostrar_info_modelos()

    # Processar cubagem
    resultado_cubagem = processar_cubagem()

    if resultado_cubagem is None:
        return

    volumes_arvore, stats_cubagem = resultado_cubagem

    # Botão para executar análise
    if st.button("🚀 Executar Análise Volumétrica", type="primary", use_container_width=True):
        executar_analise_volumetrica(volumes_arvore)

    # Mostrar resultados salvos se existirem
    if st.session_state.get('resultados_volumetricos'):
        resultados_salvos = st.session_state.resultados_volumetricos
        mostrar_resultados_volumetricos(
            resultados_salvos['resultados'],
            resultados_salvos['predicoes'],
            resultados_salvos['volumes'],
            "salvos"
        )


if __name__ == "__main__":
    main()