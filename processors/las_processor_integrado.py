# processors/las_processor_integrado.py
"""
Processador integrado de arquivos LAS/LAZ para o Sistema GreenVista
Versão melhorada com processamento em chunks, gestão de memória e interface completa
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import gc
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Callable
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Bibliotecas para processamento LAS
try:
    import laspy
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from scipy import stats

    LIDAR_DISPONIVEL = True
except ImportError:
    LIDAR_DISPONIVEL = False

from utils.formatacao import formatar_brasileiro, formatar_numero_inteligente


class ProcessadorLASIntegrado:
    """
    Classe principal para processamento integrado de arquivos LAS/LAZ
    com otimização para arquivos grandes e gestão automática de memória
    """

    def __init__(self):
        # Configurações de processamento
        self.chunk_size = 500_000  # 500K pontos por chunk (reduzido para melhor gestão de memória)
        self.min_points_per_plot = 5  # Mínimo de pontos por parcela
        self.height_threshold = 1.3  # Altura mínima para análise (metros)
        self.max_height = 80.0  # Altura máxima realística
        self.buffer_parcela = 11.28  # Raio padrão da parcela (400m²)

        # Configurações de qualidade
        self.max_file_size_mb = 500  # Tamanho máximo do arquivo em MB
        self.max_points_total = 50_000_000  # Máximo de pontos para processar

        # Estado do processamento
        self.estado_processamento = {
            'iniciado': False,
            'concluido': False,
            'erro': None,
            'progresso': 0,
            'etapa_atual': '',
            'total_pontos': 0,
            'pontos_processados': 0,
            'memoria_usada_mb': 0
        }

    def verificar_disponibilidade(self) -> Tuple[bool, List[str]]:
        """
        Verifica se as bibliotecas LiDAR estão disponíveis

        Returns:
            Tuple[bool, List[str]]: (disponível, lista de dependências faltantes)
        """
        dependencias_faltantes = []

        if not LIDAR_DISPONIVEL:
            dependencias_faltantes.extend([
                "laspy", "geopandas", "shapely", "scipy"
            ])

        return len(dependencias_faltantes) == 0, dependencias_faltantes

    def _pontos_na_parcela_otimizado(self, x_coords, y_coords, parcela_info) -> np.ndarray:
        """Versão otimizada para verificar pontos dentro de parcela"""
        try:
            geometry = parcela_info['geometry']

            # Método otimizado usando bounding box primeiro
            if hasattr(geometry, 'bounds'):
                minx, miny, maxx, maxy = geometry.bounds

                # Filtro rápido por bounding box
                bbox_mask = (
                        (x_coords >= minx) & (x_coords <= maxx) &
                        (y_coords >= miny) & (y_coords <= maxy)
                )

                if not np.any(bbox_mask):
                    return np.zeros(len(x_coords), dtype=bool)

                # Verificação precisa apenas para pontos na bbox
                pontos_bbox = np.column_stack((x_coords[bbox_mask], y_coords[bbox_mask]))

                # Usar vetorização do shapely se disponível
                from shapely.vectorized import contains
                within_geometry = contains(geometry, pontos_bbox[:, 0], pontos_bbox[:, 1])

                # Mapear de volta para array completo
                resultado = np.zeros(len(x_coords), dtype=bool)
                resultado[bbox_mask] = within_geometry

                return resultado

            else:
                # Fallback para método anterior
                return self._pontos_na_parcela_fallback(x_coords, y_coords, parcela_info)

        except Exception:
            return self._pontos_na_parcela_fallback(x_coords, y_coords, parcela_info)

    def _pontos_na_parcela_fallback(self, x_coords, y_coords, parcela_info) -> np.ndarray:
        """Método fallback para verificar pontos na parcela"""
        try:
            # Método baseado em distância ao centro
            centro = parcela_info.get('centro')
            if centro is None:
                geometry = parcela_info['geometry']
                centro = geometry.centroid

            # Calcular distâncias
            distancias = np.sqrt(
                (x_coords - centro.x) ** 2 +
                (y_coords - centro.y) ** 2
            )

            # Usar raio da parcela
            return distancias <= self.buffer_parcela

        except Exception:
            # Último recurso: retornar array vazio
            return np.zeros(len(x_coords), dtype=bool)

    def _calcular_metricas_finais(self, resultados_parcelas: Dict, interface_streamlit: bool) -> pd.DataFrame:
        """Calcula métricas finais LiDAR para cada parcela"""
        try:
            metricas_lista = []

            if interface_streamlit:
                progress_bar = st.progress(0)
                status_text = st.empty()

            total_parcelas = len(resultados_parcelas)

            for i, (parcela_id, dados) in enumerate(resultados_parcelas.items()):
                try:
                    # Extrair informações da parcela
                    if '_' in parcela_id:
                        partes = parcela_id.split('_')
                        if partes[0] == 'auto':
                            talhao, parcela = 1, int(partes[1]) + 1
                        else:
                            talhao, parcela = int(partes[0]), int(partes[1])
                    else:
                        talhao, parcela = 1, i + 1

                    # Calcular métricas da parcela
                    metricas_parcela = self._calcular_metricas_parcela(dados, talhao, parcela)

                    if metricas_parcela:
                        metricas_lista.append(metricas_parcela)

                    # Atualizar progresso
                    if interface_streamlit and i % 10 == 0:
                        progresso = i / total_parcelas
                        progress_bar.progress(progresso)
                        status_text.text(f"Calculando métricas: {i + 1}/{total_parcelas}")

                except Exception as e:
                    if interface_streamlit:
                        st.warning(f"⚠️ Erro na parcela {parcela_id}: {e}")
                    continue

            if interface_streamlit:
                progress_bar.progress(1.0)
                status_text.text("Métricas calculadas!")

            # Criar DataFrame
            if metricas_lista:
                df_metricas = pd.DataFrame(metricas_lista)

                # Ordenar por talhao e parcela
                df_metricas = df_metricas.sort_values(['talhao', 'parcela']).reset_index(drop=True)

                if interface_streamlit:
                    st.success(f"✅ Métricas calculadas para {len(df_metricas)} parcelas")

                return df_metricas
            else:
                if interface_streamlit:
                    st.error("❌ Nenhuma métrica válida calculada")
                return pd.DataFrame()

        except Exception as e:
            if interface_streamlit:
                st.error(f"❌ Erro no cálculo das métricas: {e}")
            return pd.DataFrame()

    def _calcular_metricas_parcela(self, dados_parcela: Dict, talhao: int, parcela: int) -> Optional[Dict]:
        """Calcula métricas LiDAR para uma parcela específica"""
        try:
            z_points = np.array(dados_parcela['z_points'])

            # Verificar se há pontos suficientes
            if len(z_points) < self.min_points_per_plot:
                return None

            # Filtrar alturas válidas
            z_valid = z_points[
                (z_points >= self.height_threshold) &
                (z_points <= self.max_height)
                ]

            if len(z_valid) < self.min_points_per_plot:
                return None

            # Métricas básicas de altura
            metricas = {
                'talhao': talhao,
                'parcela': parcela,
                'n_pontos': len(z_points),
                'n_pontos_validos': len(z_valid),
                'altura_media': float(np.mean(z_valid)),
                'altura_maxima': float(np.max(z_valid)),
                'altura_minima': float(np.min(z_valid)),
                'desvio_altura': float(np.std(z_valid)),
                'altura_p95': float(np.percentile(z_valid, 95)),
                'altura_p75': float(np.percentile(z_valid, 75)),
                'altura_p50': float(np.percentile(z_valid, 50)),
                'altura_p25': float(np.percentile(z_valid, 25))
            }

            # Métricas de distribuição
            metricas['cv_altura'] = (metricas['desvio_altura'] / metricas['altura_media']) * 100
            metricas['amplitude_altura'] = metricas['altura_maxima'] - metricas['altura_minima']

            # Densidade de pontos (pontos por m²)
            area_parcela = np.pi * (self.buffer_parcela ** 2)  # Área circular
            metricas['densidade_pontos'] = len(z_points) / area_parcela

            # Cobertura do dossel (% pontos acima de threshold)
            pontos_dossel = np.sum(z_valid >= self.height_threshold)
            metricas['cobertura'] = (pontos_dossel / len(z_points)) * 100

            # Métricas de complexidade estrutural
            if len(z_valid) > 10:
                # Rugosidade (desvio da superfície)
                metricas['rugosidade'] = float(np.std(np.diff(np.sort(z_valid))))

                # Índice de Shannon para diversidade de alturas
                bins = np.histogram(z_valid, bins=10)[0]
                bins = bins[bins > 0]  # Remove bins vazios
                if len(bins) > 1:
                    p = bins / np.sum(bins)
                    metricas['shannon_height'] = float(-np.sum(p * np.log(p)))
                else:
                    metricas['shannon_height'] = 0.0
            else:
                metricas['rugosidade'] = 0.0
                metricas['shannon_height'] = 0.0

            # Métricas de intensidade se disponível
            i_points = dados_parcela.get('i_points', [])
            if i_points and len(i_points) > 0:
                i_array = np.array(i_points)
                i_valid = i_array[i_array > 0]

                if len(i_valid) > 0:
                    metricas['intensidade_media'] = float(np.mean(i_valid))
                    metricas['intensidade_desvio'] = float(np.std(i_valid))
                else:
                    metricas['intensidade_media'] = 0.0
                    metricas['intensidade_desvio'] = 0.0
            else:
                metricas['intensidade_media'] = None
                metricas['intensidade_desvio'] = None

            # Métricas de retorno se disponível
            r_points = dados_parcela.get('r_points', [])
            if r_points and len(r_points) > 0:
                r_array = np.array(r_points)

                # Proporção de primeiros retornos
                primeiros_retornos = np.sum(r_array == 1)
                metricas['prop_primeiro_retorno'] = (primeiros_retornos / len(r_array)) * 100

                # Número médio de retornos
                metricas['retornos_medio'] = float(np.mean(r_array))
            else:
                metricas['prop_primeiro_retorno'] = None
                metricas['retornos_medio'] = None

            return metricas

        except Exception as e:
            # Log do erro sem quebrar o processamento
            return None

    def _limpar_memoria_intermediaria(self):
        """Limpa memória durante processamento"""
        import gc
        gc.collect()

    def _limpar_memoria(self):
        """Limpeza completa de memória"""
        import gc
        # Forçar múltiplas passadas de garbage collection
        for _ in range(3):
            gc.collect()

    def _estimar_memoria_usada(self) -> float:
        """Estima uso atual de memória em MB"""
        try:
            import psutil
            processo = psutil.Process()
            return processo.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def _otimizar_resultados_parciais(self, resultados_parcelas: Dict):
        """Otimiza resultados parciais para economizar memória"""
        try:
            # Converter listas longas para arrays numpy e depois de volta para listas menores
            for parcela_id, dados in resultados_parcelas.items():
                for key in ['z_points', 'i_points', 'r_points']:
                    if key in dados and len(dados[key]) > 10000:
                        # Manter apenas uma amostra para economizar memória
                        array_dados = np.array(dados[key])
                        if len(array_dados) > 50000:
                            # Fazer amostragem se muito grande
                            indices = np.random.choice(len(array_dados), 50000, replace=False)
                            dados[key] = array_dados[indices].tolist()
        except Exception:
            pass  # Não falhar se otimização der problema

    def _finalizar_processamento(self, metricas_finais, interface_streamlit: bool):
        """Finaliza processamento e atualiza estado"""
        self.estado_processamento.update({
            'concluido': True,
            'progresso': 100,
            'etapa_atual': 'Concluído',
            'timestamp_fim': datetime.now()
        })

        if metricas_finais is not None and not metricas_finais.empty:
            if interface_streamlit:
                st.success(f"🎉 Processamento LAS concluído: {len(metricas_finais)} parcelas processadas")
        else:
            self.estado_processamento['erro'] = 'Nenhuma métrica válida gerada'

    def _mostrar_instrucoes_instalacao(self, deps_faltantes: List[str]):
        """Mostra instruções de instalação para dependências"""
        st.error("📦 Dependências LiDAR não instaladas")

        with st.expander("📋 Instruções de Instalação"):
            st.markdown("**Instale as bibliotecas necessárias:**")

            comandos = [
                "pip install laspy[lazrs,laszip]",
                "pip install geopandas",
                "pip install shapely",
                "pip install scipy"
            ]

            for cmd in comandos:
                st.code(cmd)

            st.markdown("**Ou instale todas de uma vez:**")
            st.code("pip install laspy[lazrs,laszip] geopandas shapely scipy")

            st.info("💡 Após a instalação, reinicie o Streamlit")

    def obter_estado_processamento(self) -> Dict:
        """Retorna estado atual do processamento"""
        return self.estado_processamento.copy()

    def resetar_estado(self):
        """Reseta estado do processamento"""
        self.estado_processamento = {
            'iniciado': False,
            'concluido': False,
            'erro': None,
            'progresso': 0,
            'etapa_atual': '',
            'total_pontos': 0,
            'pontos_processados': 0,
            'memoria_usada_mb': 0
        }


# Funções utilitárias para integração com o sistema

def criar_interface_processamento_las():
    """
    Cria interface completa para processamento de arquivos LAS/LAZ
    Para uso na página LiDAR
    """
    st.subheader("🛩️ Processamento Direto de Arquivos LAS/LAZ")

    # Informações sobre o processamento
    with st.expander("ℹ️ Sobre o Processamento LAS/LAZ"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **📁 Formatos Suportados:**
            - Arquivos .LAS (padrão)
            - Arquivos .LAZ (comprimido)
            - Até 500MB por arquivo
            - Até 50 milhões de pontos

            **🔧 Processamento:**
            - Chunks automáticos para arquivos grandes
            - Gestão inteligente de memória
            - Processamento paralelo de parcelas
            """)

        with col2:
            st.markdown("""
            **📊 Métricas Calculadas:**
            - Alturas (média, máxima, percentis)
            - Variabilidade estrutural
            - Densidade de pontos
            - Cobertura do dossel
            - Complexidade estrutural
            - Intensidade (se disponível)
            """)

    return True


def criar_configuracoes_processamento_las():
    """
    Cria interface para configurações de processamento LAS
    """
    st.subheader("⚙️ Configurações de Processamento")

    col1, col2, col3 = st.columns(3)

    with col1:
        chunk_size = st.number_input(
            "📦 Tamanho do Chunk",
            min_value=100_000,
            max_value=2_000_000,
            value=500_000,
            step=100_000,
            help="Número de pontos por chunk (menor = menos memória)"
        )

        min_points = st.number_input(
            "📍 Pontos Mínimos/Parcela",
            min_value=1,
            max_value=50,
            value=5,
            help="Mínimo de pontos para considerar parcela válida"
        )

    with col2:
        height_threshold = st.number_input(
            "📏 Altura Mínima (m)",
            min_value=0.1,
            max_value=5.0,
            value=1.3,
            step=0.1,
            help="Altura mínima para análise do dossel"
        )

        max_height = st.number_input(
            "📏 Altura Máxima (m)",
            min_value=20.0,
            max_value=150.0,
            value=80.0,
            step=5.0,
            help="Altura máxima realística"
        )

    with col3:
        buffer_parcela = st.number_input(
            "🎯 Raio da Parcela (m)",
            min_value=5.0,
            max_value=50.0,
            value=11.28,
            step=0.1,
            help="Raio da parcela circular (11.28m = 400m²)"
        )

        area_calculada = np.pi * (buffer_parcela ** 2)
        st.metric("📐 Área da Parcela", f"{area_calculada:.0f} m²")

    return {
        'chunk_size': int(chunk_size),
        'min_points_per_plot': int(min_points),
        'height_threshold': height_threshold,
        'max_height': max_height,
        'buffer_parcela': buffer_parcela
    }


def processar_las_com_interface(arquivo_las, parcelas_inventario=None):
    """
    Função principal para processar arquivo LAS com interface Streamlit completa

    Args:
        arquivo_las: Arquivo LAS/LAZ carregado
        parcelas_inventario: DataFrame com parcelas (opcional)

    Returns:
        DataFrame com métricas LiDAR ou None
    """
    try:
        # Criar processador
        processador = ProcessadorLASIntegrado()

        # Verificar disponibilidade
        disponivel, deps_faltantes = processador.verificar_disponibilidade()
        if not disponivel:
            st.error("❌ Dependências LiDAR não disponíveis")
            processador._mostrar_instrucoes_instalacao(deps_faltantes)
            return None

        # Configurações de processamento
        st.subheader("⚙️ Configurações")
        config = criar_configuracoes_processamento_las()

        # Botão para iniciar processamento
        if st.button("🚀 Processar Arquivo LAS/LAZ", type="primary", use_container_width=True):
            # Mostrar informações do arquivo
            st.info(f"📁 Processando: {arquivo_las.name} ({arquivo_las.size / (1024 * 1024):.1f} MB)")

            # Container para status do processamento
            status_container = st.container()

            with status_container:
                # Callback para atualizar progresso
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                def callback_progresso(porcentagem, mensagem):
                    progress_placeholder.progress(porcentagem / 100)
                    status_placeholder.info(f"🔄 {mensagem}")

                # Executar processamento
                resultado = processador.processar_arquivo_las_completo(
                    arquivo_las=arquivo_las,
                    parcelas_inventario=parcelas_inventario,
                    config_processamento=config,
                    callback_progresso=callback_progresso,
                    interface_streamlit=True
                )

                # Limpar placeholders de progresso
                progress_placeholder.empty()
                status_placeholder.empty()

                return resultado

        return None

    except Exception as e:
        st.error(f"❌ Erro no processamento: {e}")
        return None


def mostrar_resultados_processamento_las(df_metricas):
    """
    Mostra resultados do processamento LAS de forma organizada

    Args:
        df_metricas: DataFrame com métricas LiDAR calculadas
    """
    if df_metricas is None or df_metricas.empty:
        st.warning("⚠️ Nenhuma métrica LiDAR disponível")
        return

    st.subheader("📊 Resultados do Processamento LAS")

    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📍 Parcelas Processadas", len(df_metricas))

    with col2:
        altura_media_geral = df_metricas['altura_media'].mean()
        st.metric("🌳 Altura Média Geral", f"{altura_media_geral:.1f} m")

    with col3:
        pontos_total = df_metricas['n_pontos'].sum()
        st.metric("📊 Total de Pontos", f"{pontos_total:,}")

    with col4:
        cobertura_media = df_metricas['cobertura'].mean()
        st.metric("🍃 Cobertura Média", f"{cobertura_media:.1f}%")

    # Tabela de resultados
    st.subheader("📋 Métricas por Parcela")

    # Selecionar colunas principais para exibição
    colunas_exibir = [
        'talhao', 'parcela', 'n_pontos', 'altura_media', 'altura_maxima',
        'desvio_altura', 'cobertura', 'densidade_pontos'
    ]

    # Filtrar apenas colunas existentes
    colunas_existentes = [col for col in colunas_exibir if col in df_metricas.columns]

    # Formatar dados para exibição
    df_display = df_metricas[colunas_existentes].copy()

    # Renomear colunas para português
    renome_colunas = {
        'talhao': 'Talhão',
        'parcela': 'Parcela',
        'n_pontos': 'N° Pontos',
        'altura_media': 'Altura Média (m)',
        'altura_maxima': 'Altura Máx (m)',
        'desvio_altura': 'Desvio Alt (m)',
        'cobertura': 'Cobertura (%)',
        'densidade_pontos': 'Densidade (pts/m²)'
    }

    df_display = df_display.rename(columns=renome_colunas)

    # Formatar números
    for col in df_display.columns:
        if col in ['Altura Média (m)', 'Altura Máx (m)', 'Desvio Alt (m)']:
            df_display[col] = df_display[col].round(1)
        elif col in ['Cobertura (%)', 'Densidade (pts/m²)']:
            df_display[col] = df_display[col].round(2)

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Download dos resultados
    st.subheader("💾 Download dos Resultados")

    col1, col2 = st.columns(2)

    with col1:
        # CSV das métricas
        csv_metricas = df_metricas.to_csv(index=False, sep=';')
        st.download_button(
            "📥 Download Métricas CSV",
            csv_metricas,
            "metricas_lidar_las.csv",
            "text/csv",
            help="Download das métricas em formato CSV"
        )

    with col2:
        # Excel das métricas
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_metricas.to_excel(writer, sheet_name='Metricas_LiDAR', index=False)

        st.download_button(
            "📊 Download Métricas Excel",
            buffer.getvalue(),
            "metricas_lidar_las.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download das métricas em formato Excel"
        )


# Função para integração com a página LiDAR existente
def integrar_com_pagina_lidar():
    """
    Função para integrar processamento LAS com página LiDAR existente
    Retorna True se processamento LAS está disponível
    """
    processador = ProcessadorLASIntegrado()
    disponivel, _ = processador.verificar_disponibilidade()
    return disponivel

  # Configurações de processamento
    self.chunk_size = 500_000  # 500K pontos por chunk (reduzido para melhor gestão de memória)
    self.min_points_per_plot = 5  # Mínimo de pontos por parcela
    self.height_threshold = 1.3  # Altura mínima para análise (metros)
    self.max_height = 80.0  # Altura máxima realística
    self.buffer_parcela = 11.28  # Raio padrão da parcela (400m²)

    # Configurações de qualidade
    self.max_file_size_mb = 500  # Tamanho máximo do arquivo em MB
    self.max_points_total = 50_000_000  # Máximo de pontos para processar

    # Estado do processamento
    self.estado_processamento = {
        'iniciado': False,
        'concluido': False,
        'erro': None,
        'progresso': 0,
        'etapa_atual': '',
        'total_pontos': 0,
        'pontos_processados': 0,
        'memoria_usada_mb': 0
    }


def verificar_disponibilidade(self) -> Tuple[bool, List[str]]:
    """
    Verifica se as bibliotecas LiDAR estão disponíveis

    Returns:
        Tuple[bool, List[str]]: (disponível, lista de dependências faltantes)
    """
    dependencias_faltantes = []

    if not LIDAR_DISPONIVEL:
        dependencias_faltantes.extend([
            "laspy", "geopandas", "shapely", "scipy"
        ])

    return len(dependencias_faltantes) == 0, dependencias_faltantes


def processar_arquivo_las_completo(
        self,
        arquivo_las,
        parcelas_inventario: Optional[pd.DataFrame] = None,
        config_processamento: Optional[Dict] = None,
        callback_progresso: Optional[Callable] = None,
        interface_streamlit: bool = True
) -> Optional[pd.DataFrame]:
    """
    Método principal para processamento completo de arquivo LAS/LAZ

    Args:
        arquivo_las: Arquivo LAS/LAZ carregado
        parcelas_inventario: DataFrame com parcelas do inventário
        config_processamento: Configurações customizadas
        callback_progresso: Função callback para progresso
        interface_streamlit: Se deve usar interface do Streamlit

    Returns:
        DataFrame com métricas LiDAR ou None se erro
    """
    try:
        # Verificar disponibilidade
        disponivel, deps_faltantes = self.verificar_disponibilidade()
        if not disponivel:
            erro_msg = f"Bibliotecas LiDAR não disponíveis: {', '.join(deps_faltantes)}"
            if interface_streamlit:
                st.error(f"❌ {erro_msg}")
                self._mostrar_instrucoes_instalacao(deps_faltantes)
            self.estado_processamento['erro'] = erro_msg
            return None

        # Aplicar configurações customizadas
        if config_processamento:
            self._aplicar_configuracoes(config_processamento)

        # Inicializar estado
        self._inicializar_estado_processamento()

        # Validar arquivo
        if not self._validar_arquivo_upload(arquivo_las, interface_streamlit):
            return None

        # Processar arquivo LAS
        with self._gerenciar_arquivo_temporario(arquivo_las) as temp_path:
            if temp_path is None:
                return None

            # Carregar e validar arquivo LAS
            las_file = self._carregar_arquivo_las(temp_path, callback_progresso, interface_streamlit)
            if las_file is None:
                return None

            # Processar baseado no tamanho
            metricas_finais = self._processar_por_tamanho(
                las_file, parcelas_inventario, callback_progresso, interface_streamlit
            )

            # Finalizar processamento
            self._finalizar_processamento(metricas_finais, interface_streamlit)

            return metricas_finais

    except Exception as e:
        erro_msg = f"Erro no processamento LAS: {str(e)}"
        self.estado_processamento['erro'] = erro_msg

        if interface_streamlit:
            st.error(f"❌ {erro_msg}")
            with st.expander("🔍 Detalhes do erro"):
                st.code(str(e))

        return None
    finally:
        # Limpar memória
        self._limpar_memoria()


def _aplicar_configuracoes(self, config: Dict):
    """Aplica configurações customizadas"""
    self.chunk_size = config.get('chunk_size', self.chunk_size)
    self.min_points_per_plot = config.get('min_points_per_plot', self.min_points_per_plot)
    self.height_threshold = config.get('height_threshold', self.height_threshold)
    self.max_height = config.get('max_height', self.max_height)
    self.buffer_parcela = config.get('buffer_parcela', self.buffer_parcela)


def _inicializar_estado_processamento(self):
    """Inicializa estado do processamento"""
    self.estado_processamento.update({
        'iniciado': True,
        'concluido': False,
        'erro': None,
        'progresso': 0,
        'etapa_atual': 'Inicializando...',
        'total_pontos': 0,
        'pontos_processados': 0,
        'memoria_usada_mb': 0,
        'timestamp_inicio': datetime.now()
    })


def _validar_arquivo_upload(self, arquivo_las, interface_streamlit: bool) -> bool:
    """Valida arquivo antes do processamento"""
    try:
        # Verificar tamanho do arquivo
        tamanho_mb = arquivo_las.size / (1024 * 1024)

        if tamanho_mb > self.max_file_size_mb:
            erro_msg = f"Arquivo muito grande: {tamanho_mb:.1f}MB (máximo: {self.max_file_size_mb}MB)"
            if interface_streamlit:
                st.error(f"❌ {erro_msg}")
            self.estado_processamento['erro'] = erro_msg
            return False

        # Verificar extensão
        nome_arquivo = arquivo_las.name.lower()
        if not (nome_arquivo.endswith('.las') or nome_arquivo.endswith('.laz')):
            erro_msg = f"Formato não suportado: {nome_arquivo}. Use .las ou .laz"
            if interface_streamlit:
                st.error(f"❌ {erro_msg}")
            self.estado_processamento['erro'] = erro_msg
            return False

        if interface_streamlit:
            st.info(f"📁 Arquivo validado: {arquivo_las.name} ({tamanho_mb:.1f}MB)")

        return True

    except Exception as e:
        if interface_streamlit:
            st.error(f"❌ Erro na validação: {e}")
        return False


def _gerenciar_arquivo_temporario(self, arquivo_las):
    """Context manager para arquivo temporário"""

    class GerenciadorArquivoTemp:
        def __init__(self, arquivo):
            self.arquivo = arquivo
            self.temp_path = None

        def __enter__(self):
            try:
                # Criar arquivo temporário
                suffix = '.laz' if self.arquivo.name.lower().endswith('.laz') else '.las'
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_file.write(self.arquivo.read())
                    self.temp_path = temp_file.name

                # Reset do arquivo original para não interferir em outros usos
                self.arquivo.seek(0)

                return self.temp_path
            except Exception as e:
                st.error(f"❌ Erro ao criar arquivo temporário: {e}")
                return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Limpar arquivo temporário
            if self.temp_path and os.path.exists(self.temp_path):
                try:
                    os.unlink(self.temp_path)
                except Exception:
                    pass  # Não falhar se não conseguir deletar

    return GerenciadorArquivoTemp(arquivo_las)


def _carregar_arquivo_las(self, temp_path: str, callback_progresso, interface_streamlit: bool):
    """Carrega e valida arquivo LAS"""
    try:
        self.estado_processamento['etapa_atual'] = 'Carregando arquivo LAS...'

        if callback_progresso:
            callback_progresso(5, "Carregando arquivo LAS...")
        elif interface_streamlit:
            with st.spinner("📡 Carregando arquivo LAS/LAZ..."):
                time.sleep(0.1)  # Para mostrar o spinner

        # Carregar arquivo
        las_file = laspy.read(temp_path)

        # Validar estrutura
        if not self._validar_estrutura_las(las_file, interface_streamlit):
            return None

        # Atualizar estado
        self.estado_processamento['total_pontos'] = len(las_file.points)

        if callback_progresso:
            callback_progresso(10, f"Arquivo carregado: {len(las_file.points):,} pontos")

        return las_file

    except Exception as e:
        erro_msg = f"Erro ao carregar arquivo LAS: {e}"
        if interface_streamlit:
            st.error(f"❌ {erro_msg}")
        self.estado_processamento['erro'] = erro_msg
        return None


def _validar_estrutura_las(self, las_file, interface_streamlit: bool) -> bool:
    """Valida estrutura do arquivo LAS"""
    try:
        # Verificar se tem pontos
        if len(las_file.points) == 0:
            if interface_streamlit:
                st.error("❌ Arquivo LAS vazio")
            return False

        # Verificar coordenadas
        if not all(hasattr(las_file, attr) for attr in ['x', 'y', 'z']):
            if interface_streamlit:
                st.error("❌ Arquivo LAS sem coordenadas XYZ")
            return False

        # Verificar limite de pontos
        if len(las_file.points) > self.max_points_total:
            if interface_streamlit:
                st.error(f"❌ Muitos pontos: {len(las_file.points):,} (máximo: {self.max_points_total:,})")
            return False

        # Validar alturas
        z_values = las_file.z
        z_min, z_max = np.min(z_values), np.max(z_values)

        if z_max - z_min > 1000:
            if interface_streamlit:
                st.warning("⚠️ Diferenças de altura muito grandes - pode não estar normalizado")

        if z_max > 10000:
            if interface_streamlit:
                st.warning("⚠️ Alturas muito altas - arquivo pode não estar normalizado")

        # Feedback de sucesso
        if interface_streamlit:
            st.success(f"✅ Arquivo LAS válido: {len(las_file.points):,} pontos")
            st.info(f"📊 Extensão Z: {z_min:.1f}m a {z_max:.1f}m")

        return True

    except Exception as e:
        if interface_streamlit:
            st.error(f"❌ Erro na validação: {e}")
        return False


def _processar_por_tamanho(self, las_file, parcelas_inventario, callback_progresso, interface_streamlit):
    """Decide estratégia de processamento baseado no tamanho do arquivo"""
    total_points = len(las_file.points)

    if total_points > self.chunk_size:
        if interface_streamlit:
            st.info(f"📊 Arquivo grande detectado: processamento em chunks")

        return self._processar_arquivo_grande_melhorado(
            las_file, parcelas_inventario, callback_progresso, interface_streamlit
        )
    else:
        if interface_streamlit:
            st.info(f"📊 Arquivo pequeno: processamento direto")

        return self._processar_arquivo_pequeno_melhorado(
            las_file, parcelas_inventario, callback_progresso, interface_streamlit
        )


def _processar_arquivo_grande_melhorado(self, las_file, parcelas_inventario, callback_progresso, interface_streamlit):
    """Processamento otimizado para arquivos grandes"""
    total_points = len(las_file.points)
    num_chunks = (total_points + self.chunk_size - 1) // self.chunk_size

    self.estado_processamento['etapa_atual'] = f'Processando {num_chunks} chunks...'

    if interface_streamlit:
        st.info(f"📊 Processando {total_points:,} pontos em {num_chunks} chunks de {self.chunk_size:,}")

    # Definir parcelas
    parcelas = self._criar_parcelas(las_file, parcelas_inventario, interface_streamlit)
    if not parcelas:
        return None

    # Inicializar containers de resultados
    resultados_parcelas = {}

    # Barra de progresso para interface Streamlit
    if interface_streamlit:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Processar cada chunk
        for chunk_idx in range(num_chunks):
            # Calcular índices do chunk
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_points)

            # Atualizar status
            self.estado_processamento['etapa_atual'] = f'Chunk {chunk_idx + 1}/{num_chunks}'
            self.estado_processamento['pontos_processados'] = end_idx

            # Extrair dados do chunk
            chunk_data = self._extrair_chunk_dados(las_file, start_idx, end_idx)

            # Processar chunk para cada parcela
            self._processar_chunk_parcelas(chunk_data, parcelas, resultados_parcelas)

            # Atualizar progresso
            progresso_pct = 20 + int(60 * (chunk_idx + 1) / num_chunks)
            self.estado_processamento['progresso'] = progresso_pct

            if callback_progresso:
                callback_progresso(progresso_pct, f"Chunk {chunk_idx + 1}/{num_chunks} processado")

            if interface_streamlit:
                progress_bar.progress(progresso_pct / 100)
                status_text.text(f"Processando chunk {chunk_idx + 1}/{num_chunks}")

            # Gestão de memória
            if chunk_idx % 3 == 0:  # A cada 3 chunks
                self._limpar_memoria_intermediaria()

            # Verificar uso de memória
            memoria_mb = self._estimar_memoria_usada()
            self.estado_processamento['memoria_usada_mb'] = memoria_mb

            if memoria_mb > 1000:  # Se usar mais de 1GB
                if interface_streamlit:
                    st.warning("⚠️ Alto uso de memória - otimizando...")
                self._otimizar_resultados_parciais(resultados_parcelas)

        # Calcular métricas finais
        if callback_progresso:
            callback_progresso(90, "Calculando métricas finais...")

        if interface_streamlit:
            progress_bar.progress(0.9)
            status_text.text("Calculando métricas finais...")

        metricas_finais = self._calcular_metricas_finais(resultados_parcelas, interface_streamlit)

        if interface_streamlit:
            progress_bar.progress(1.0)
            status_text.text("Processamento concluído!")

        return metricas_finais

    except Exception as e:
        erro_msg = f"Erro no processamento em chunks: {e}"
        self.estado_processamento['erro'] = erro_msg
        if interface_streamlit:
            st.error(f"❌ {erro_msg}")
        return None

    finally:
        # Limpeza final
        self._limpar_memoria()


def _processar_arquivo_pequeno_melhorado(self, las_file, parcelas_inventario, callback_progresso, interface_streamlit):
    """Processamento otimizado para arquivos pequenos"""
    try:
        # Extrair dados
        x_coords = las_file.x
        y_coords = las_file.y
        z_coords = las_file.z

        # Atributos opcionais
        intensity = getattr(las_file, 'intensity', None)
        return_number = getattr(las_file, 'return_number', None)

        if callback_progresso:
            callback_progresso(30, "Definindo parcelas...")

        # Definir parcelas
        parcelas = self._criar_parcelas(las_file, parcelas_inventario, interface_streamlit)
        if not parcelas:
            return None

        if callback_progresso:
            callback_progresso(50, "Extraindo pontos por parcela...")

        # Processar cada parcela
        resultados_parcelas = {}
        total_parcelas = len(parcelas)

        if interface_streamlit:
            progress_bar = st.progress(0)
            status_text = st.empty()

        for i, (parcela_id, parcela_info) in enumerate(parcelas.items()):
            # Verificar pontos na parcela
            pontos_na_parcela = self._pontos_na_parcela_otimizado(
                x_coords, y_coords, parcela_info
            )

            if np.any(pontos_na_parcela):
                # Extrair dados da parcela
                dados_parcela = {
                    'z_points': z_coords[pontos_na_parcela].tolist(),
                    'i_points': intensity[pontos_na_parcela].tolist() if intensity is not None else [],
                    'r_points': return_number[pontos_na_parcela].tolist() if return_number is not None else []
                }

                resultados_parcelas[parcela_id] = dados_parcela

            # Atualizar progresso
            if i % 5 == 0:  # Atualizar a cada 5 parcelas
                progresso = 50 + int(30 * i / total_parcelas)

                if callback_progresso:
                    callback_progresso(progresso, f"Parcela {i + 1}/{total_parcelas}")

                if interface_streamlit:
                    progress_bar.progress(progresso / 100)
                    status_text.text(f"Processando parcela {i + 1}/{total_parcelas}")

        # Calcular métricas
        if callback_progresso:
            callback_progresso(90, "Calculando métricas...")

        if interface_streamlit:
            progress_bar.progress(0.9)
            status_text.text("Calculando métricas...")

        metricas_finais = self._calcular_metricas_finais(resultados_parcelas, interface_streamlit)

        if interface_streamlit:
            progress_bar.progress(1.0)
            status_text.text("Processamento concluído!")

        return metricas_finais

    except Exception as e:
        erro_msg = f"Erro no processamento direto: {e}"
        self.estado_processamento['erro'] = erro_msg
        if interface_streamlit:
            st.error(f"❌ {erro_msg}")
        return None


def _extrair_chunk_dados(self, las_file, start_idx: int, end_idx: int) -> Dict:
    """Extrai dados de um chunk específico"""
    return {
        'x': las_file.x[start_idx:end_idx],
        'y': las_file.y[start_idx:end_idx],
        'z': las_file.z[start_idx:end_idx],
        'intensity': getattr(las_file, 'intensity', None)[start_idx:end_idx] if hasattr(las_file,
                                                                                        'intensity') else None,
        'return_number': getattr(las_file, 'return_number', None)[start_idx:end_idx] if hasattr(las_file,
                                                                                                'return_number') else None
    }


def _processar_chunk_parcelas(self, chunk_data: Dict, parcelas: Dict, resultados_parcelas: Dict):
    """Processa chunk para todas as parcelas"""
    x_coords = chunk_data['x']
    y_coords = chunk_data['y']
    z_coords = chunk_data['z']
    intensity = chunk_data['intensity']
    return_number = chunk_data['return_number']

    for parcela_id, parcela_info in parcelas.items():
        # Verificar pontos na parcela
        pontos_na_parcela = self._pontos_na_parcela_otimizado(
            x_coords, y_coords, parcela_info
        )

        if np.any(pontos_na_parcela):
            # Extrair pontos
            z_parcela = z_coords[pontos_na_parcela]
            i_parcela = intensity[pontos_na_parcela] if intensity is not None else None
            r_parcela = return_number[pontos_na_parcela] if return_number is not None else None

            # Acumular resultados
            if parcela_id not in resultados_parcelas:
                resultados_parcelas[parcela_id] = {
                    'z_points': [],
                    'i_points': [],
                    'r_points': []
                }

            resultados_parcelas[parcela_id]['z_points'].extend(z_parcela.tolist())
            if i_parcela is not None:
                resultados_parcelas[parcela_id]['i_points'].extend(i_parcela.tolist())
            if r_parcela is not None:
                resultados_parcelas[parcela_id]['r_points'].extend(r_parcela.tolist())


def _criar_parcelas(self, las_file, parcelas_inventario, interface_streamlit: bool) -> Optional[Dict]:
    """Cria definições de parcelas para processamento"""
    try:
        if parcelas_inventario is not None and not parcelas_inventario.empty:
            if interface_streamlit:
                st.info("📍 Usando parcelas do inventário")
            return self._criar_parcelas_inventario(parcelas_inventario, las_file, interface_streamlit)
        else:
            if interface_streamlit:
                st.info("🔲 Criando grid automático")
            return self._criar_grid_automatico(las_file, interface_streamlit)

    except Exception as e:
        if interface_streamlit:
            st.error(f"❌ Erro ao criar parcelas: {e}")
        return None


def _criar_parcelas_inventario(self, parcelas_inventario: pd.DataFrame, las_file, interface_streamlit: bool) -> Dict:
    """Cria parcelas baseadas no inventário"""
    parcelas = {}

    try:
        if 'x' in parcelas_inventario.columns and 'y' in parcelas_inventario.columns:
            # Usar coordenadas fornecidas
            for idx, row in parcelas_inventario.iterrows():
                centro = Point(row['x'], row['y'])
                parcela_circular = centro.buffer(self.buffer_parcela)

                parcela_key = f"{row['talhao']}_{row['parcela']}"
                parcelas[parcela_key] = {
                    'geometry': parcela_circular,
                    'centro': centro,
                    'talhao': row['talhao'],
                    'parcela': row['parcela']
                }

            if interface_streamlit:
                st.success(f"✅ Criadas {len(parcelas)} parcelas com coordenadas")

        else:
            # Estimativa baseada na extensão do LAS
            if interface_streamlit:
                st.warning("⚠️ Sem coordenadas no inventário. Criando estimativa...")

            parcelas = self._estimar_parcelas_sem_coordenadas(parcelas_inventario, las_file)

    except Exception as e:
        if interface_streamlit:
            st.error(f"❌ Erro ao criar parcelas do inventário: {e}")
        return {}

    return parcelas


def _estimar_parcelas_sem_coordenadas(self, parcelas_inventario: pd.DataFrame, las_file) -> Dict:
    """Estima localização de parcelas quando coordenadas não estão disponíveis"""
    parcelas = {}

    # Obter extensão do arquivo LAS
    x_min, x_max = np.min(las_file.x), np.max(las_file.x)
    y_min, y_max = np.min(las_file.y), np.max(las_file.y)

    # Distribuir parcelas uniformemente
    talhoes_unicos = parcelas_inventario['talhao'].unique()
    parcelas_por_talhao = parcelas_inventario.groupby('talhao')['parcela'].nunique()

    total_parcelas = parcelas_por_talhao.sum()
    cols = int(np.ceil(np.sqrt(total_parcelas)))
    rows = int(np.ceil(total_parcelas / cols))

    # Calcular espaçamento
    x_spacing = (x_max - x_min) / cols
    y_spacing = (y_max - y_min) / rows

    parcela_idx = 0
    for talhao in talhoes_unicos:
        n_parcelas = parcelas_por_talhao[talhao]

        for parcela in range(1, n_parcelas + 1):
            col = parcela_idx % cols
            row = parcela_idx // cols

            x_centro = x_min + (col + 0.5) * x_spacing
            y_centro = y_min + (row + 0.5) * y_spacing

            centro = Point(x_centro, y_centro)
            parcela_circular = centro.buffer(self.buffer_parcela)

            parcela_key = f"{talhao}_{parcela}"
            parcelas[parcela_key] = {
                'geometry': parcela_circular,
                'centro': centro,
                'talhao': talhao,
                'parcela': parcela
            }

            parcela_idx += 1

    return parcelas


def _criar_grid_automatico(self, las_file, interface_streamlit: bool) -> Dict:
    """Cria grid automático de parcelas"""
    parcelas = {}

    try:
        # Obter extensão
        x_min, x_max = np.min(las_file.x), np.max(las_file.x)
        y_min, y_max = np.min(las_file.y), np.max(las_file.y)

        # Definir tamanho da célula (20m x 20m por padrão)
        cell_size = 20.0

        # Criar grid
        x_coords = np.arange(x_min, x_max + cell_size, cell_size)
        y_coords = np.arange(y_min, y_max + cell_size, cell_size)

        parcela_id = 0
        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                x1, x2 = x_coords[i], x_coords[i + 1]
                y1, y2 = y_coords[j], y_coords[j + 1]

                polygon = Polygon([
                    (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
                ])

                centro = Point((x1 + x2) / 2, (y1 + y2) / 2)

                parcelas[f"auto_{parcela_id}"] = {
                    'geometry': polygon,
                    'centro': centro,
                    'talhao': 1,
                    'parcela': parcela_id + 1
                }

                parcela_id += 1

        if interface_streamlit:
            st.info(f"📊 Grid automático: {len(parcelas)} células de {cell_size}m x {cell_size}m")

    except Exception as e:
        if interface_streamlit:
            st.error(f"❌ Erro ao criar grid: {e}")
        return {}

    return parcelas
