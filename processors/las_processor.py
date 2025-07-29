# processors/las_processor.py
"""
Processador nativo de arquivos LAS/LAZ para o Sistema GreenVista
Processa arquivos LiDAR diretamente no Python sem dependência do R
Otimizado para arquivos grandes com processamento em chunks
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import gc
import time
from typing import Optional, Tuple, Dict, List
import warnings

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


class ProcessadorLAS:
    """
    Classe para processamento de arquivos LAS/LAZ com otimização para arquivos grandes
    """

    def __init__(self):
        self.chunk_size = 1_000_000  # Processar 1M pontos por vez
        self.min_points_per_plot = 10  # Mínimo de pontos por parcela
        self.height_threshold = 2.0  # Altura mínima para análise (metros)
        self.max_height = 60.0  # Altura máxima realística

    def verificar_disponibilidade(self) -> bool:
        """Verifica se as bibliotecas LiDAR estão disponíveis"""
        return LIDAR_DISPONIVEL

    def processar_arquivo_las(self, arquivo_las, parcelas_inventario=None,
                              callback_progresso=None) -> Optional[pd.DataFrame]:
        """
        Processa arquivo LAS/LAZ e extrai métricas por parcela

        Args:
            arquivo_las: Arquivo LAS/LAZ carregado
            parcelas_inventario: DataFrame com parcelas do inventário
            callback_progresso: Função callback para progresso

        Returns:
            DataFrame com métricas LiDAR ou None se erro
        """
        if not self.verificar_disponibilidade():
            st.error("❌ Bibliotecas LiDAR não disponíveis. Instale: pip install laspy geopandas")
            return None

        try:
            # Salvar arquivo temporariamente se necessário
            with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as temp_file:
                temp_file.write(arquivo_las.read())
                temp_path = temp_file.name

            try:
                # Abrir arquivo LAS
                with st.spinner("📡 Carregando arquivo LAS/LAZ..."):
                    las_file = laspy.read(temp_path)

                if callback_progresso:
                    callback_progresso(10, "Arquivo carregado")

                # Validar arquivo
                if not self._validar_arquivo_las(las_file):
                    return None

                # Processar em chunks se arquivo for muito grande
                total_points = len(las_file.points)

                if total_points > self.chunk_size:
                    metricas = self._processar_arquivo_grande(
                        las_file, parcelas_inventario, callback_progresso
                    )
                else:
                    metricas = self._processar_arquivo_pequeno(
                        las_file, parcelas_inventario, callback_progresso
                    )

                return metricas

            finally:
                # Limpar arquivo temporário
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo LAS: {e}")
            return None

    def _validar_arquivo_las(self, las_file) -> bool:
        """Valida arquivo LAS/LAZ"""
        try:
            # Verificar se tem pontos
            if len(las_file.points) == 0:
                st.error("❌ Arquivo LAS vazio")
                return False

            # Verificar se tem coordenadas
            if not hasattr(las_file, 'x') or not hasattr(las_file, 'y') or not hasattr(las_file, 'z'):
                st.error("❌ Arquivo LAS sem coordenadas XYZ")
                return False

            # Verificar valores de altura razoáveis
            z_values = las_file.z
            z_min, z_max = np.min(z_values), np.max(z_values)

            if z_max - z_min > 1000:  # Diferença muito grande
                st.warning("⚠️ Diferenças de altura muito grandes no arquivo LAS")

            if z_max > 10000:  # Provavelmente não normalizado
                st.warning("⚠️ Arquivo LAS pode não estar normalizado (alturas muito altas)")

            st.success(f"✅ Arquivo LAS válido: {len(las_file.points):,} pontos")

            return True

        except Exception as e:
            st.error(f"❌ Erro na validação do arquivo LAS: {e}")
            return False

    def _processar_arquivo_grande(self, las_file, parcelas_inventario, callback_progresso) -> pd.DataFrame:
        """Processa arquivo LAS grande em chunks"""
        total_points = len(las_file.points)
        num_chunks = (total_points + self.chunk_size - 1) // self.chunk_size

        st.info(f"📊 Processando arquivo grande: {total_points:,} pontos em {num_chunks} chunks")

        # Definir parcelas se não fornecidas
        if parcelas_inventario is None:
            parcelas = self._criar_grid_automatico(las_file)
        else:
            parcelas = self._criar_parcelas_inventario(parcelas_inventario, las_file)

        # Inicializar resultados
        resultados_parcelas = {}

        # Processar cada chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_points)

            # Extrair chunk
            chunk_x = las_file.x[start_idx:end_idx]
            chunk_y = las_file.y[start_idx:end_idx]
            chunk_z = las_file.z[start_idx:end_idx]

            # Processar intensidade se disponível
            chunk_intensity = None
            if hasattr(las_file, 'intensity'):
                chunk_intensity = las_file.intensity[start_idx:end_idx]

            # Processar return number se disponível
            chunk_return_num = None
            if hasattr(las_file, 'return_number'):
                chunk_return_num = las_file.return_number[start_idx:end_idx]

            # Processar cada parcela para este chunk
            for parcela_id, parcela_geom in parcelas.items():
                # Verificar quais pontos estão na parcela
                pontos_na_parcela = self._pontos_na_parcela(
                    chunk_x, chunk_y, parcela_geom
                )

                if np.any(pontos_na_parcela):
                    # Extrair pontos da parcela
                    z_parcela = chunk_z[pontos_na_parcela]
                    i_parcela = chunk_intensity[pontos_na_parcela] if chunk_intensity is not None else None
                    r_parcela = chunk_return_num[pontos_na_parcela] if chunk_return_num is not None else None

                    # Acumular pontos para esta parcela
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

            # Atualizar progresso
            progresso = 20 + int(70 * (chunk_idx + 1) / num_chunks)
            if callback_progresso:
                callback_progresso(progresso, f"Processando chunk {chunk_idx + 1}/{num_chunks}")

            # Forçar garbage collection entre chunks
            if chunk_idx % 5 == 0:
                gc.collect()

        # Calcular métricas finais
        if callback_progresso:
            callback_progresso(90, "Calculando métricas finais...")

        metricas_finais = self._calcular_metricas_parcelas(resultados_parcelas)

        return metricas_finais

    def _processar_arquivo_pequeno(self, las_file, parcelas_inventario, callback_progresso) -> pd.DataFrame:
        """Processa arquivo LAS pequeno de uma vez"""
        # Extrair coordenadas
        x_coords = las_file.x
        y_coords = las_file.y
        z_coords = las_file.z

        # Extrair atributos opcionais
        intensity = getattr(las_file, 'intensity', None)
        return_number = getattr(las_file, 'return_number', None)

        if callback_progresso:
            callback_progresso(30, "Definindo parcelas...")

        # Definir parcelas
        if parcelas_inventario is None:
            parcelas = self._criar_grid_automatico(las_file)
        else:
            parcelas = self._criar_parcelas_inventario(parcelas_inventario, las_file)

        if callback_progresso:
            callback_progresso(50, "Extraindo pontos por parcela...")

        # Processar cada parcela
        resultados_parcelas = {}

        for i, (parcela_id, parcela_geom) in enumerate(parcelas.items()):
            # Verificar quais pontos estão na parcela
            pontos_na_parcela = self._pontos_na_parcela(x_coords, y_coords, parcela_geom)

            if np.any(pontos_na_parcela):
                z_parcela = z_coords[pontos_na_parcela]
                i_parcela = intensity[pontos_na_parcela] if intensity is not None else None
                r_parcela = return_number[pontos_na_parcela] if return_number is not None else None

                resultados_parcelas[parcela_id] = {
                    'z_points': z_parcela.tolist(),
                    'i_points': i_parcela.tolist() if i_parcela is not None else [],
                    'r_points': r_parcela.tolist() if r_parcela is not None else []
                }

            # Atualizar progresso
            if i % 10 == 0 and callback_progresso:
                progresso = 50 + int(30 * i / len(parcelas))
                callback_progresso(progresso, f"Processando parcela {i + 1}/{len(parcelas)}")

        if callback_progresso:
            callback_progresso(90, "Calculando métricas...")

        # Calcular métricas
        metricas_finais = self._calcular_metricas_parcelas(resultados_parcelas)

        return metricas_finais

    def _criar_grid_automatico(self, las_file) -> Dict:
        """Cria grid automático de parcelas baseado na extensão do arquivo"""
        # Obter extensão
        x_min, x_max = np.min(las_file.x), np.max(las_file.x)
        y_min, y_max = np.min(las_file.y), np.max(las_file.y)

        # Definir tamanho da célula (20m x 20m por padrão)
        cell_size = 20.0

        # Criar grid
        x_coords = np.arange(x_min, x_max + cell_size, cell_size)
        y_coords = np.arange(y_min, y_max + cell_size, cell_size)

        parcelas = {}
        parcela_id = 0

        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                # Criar polígono da célula
                x1, x2 = x_coords[i], x_coords[i + 1]
                y1, y2 = y_coords[j], y_coords[j + 1]

                polygon = Polygon([
                    (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
                ])

                parcelas[f"auto_{parcela_id}"] = {
                    'geometry': polygon,
                    'talhao': 1,
                    'parcela': parcela_id + 1
                }

                parcela_id += 1

        st.info(f"📊 Grid automático criado: {len(parcelas)} células de {cell_size}m x {cell_size}m")

        return parcelas

    def _criar_parcelas_inventario(self, parcelas_inventario, las_file) -> Dict:
        """Cria parcelas baseadas no inventário florestal"""
        parcelas = {}

        # Verificar se tem coordenadas no inventário
        if 'x' in parcelas_inventario.columns and 'y' in parcelas_inventario.columns:
            # Usar coordenadas fornecidas
            for idx, row in parcelas_inventario.iterrows():
                # Criar parcela circular (raio padrão 11.28m = 400m²)
                centro = Point(row['x'], row['y'])
                parcela_circular = centro.buffer(11.28)

                parcela_key = f"{row['talhao']}_{row['parcela']}"
                parcelas[parcela_key] = {
                    'geometry': parcela_circular,
                    'talhao': row['talhao'],
                    'parcela': row['parcela']
                }

        else:
            # Se não tem coordenadas, criar grid baseado nos talhões únicos
            st.warning("⚠️ Sem coordenadas no inventário. Criando parcelas estimadas...")

            # Obter extensão do arquivo LAS
            x_min, x_max = np.min(las_file.x), np.max(las_file.x)
            y_min, y_max = np.min(las_file.y), np.max(las_file.y)

            # Distribuir parcelas uniformemente
            talhoes_unicos = parcelas_inventario['talhao'].unique()
            parcelas_por_talhao = parcelas_inventario.groupby('talhao')['parcela'].nunique()

            # Calcular grid necessário
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
                    # Calcular posição no grid
                    col = parcela_idx % cols
                    row = parcela_idx // cols

                    # Coordenadas do centro
                    x_centro = x_min + (col + 0.5) * x_spacing
                    y_centro = y_min + (row + 0.5) * y_spacing

                    # Criar parcela circular
                    centro = Point(x_centro, y_centro)
                    parcela_circular = centro.buffer(11.28)

                    parcela_key = f"{talhao}_{parcela}"
                    parcelas[parcela_key] = {
                        'geometry': parcela_circular,
                        'talhao': talhao,
                        'parcela': parcela
                    }

                    parcela_idx += 1

        st.info(f"📍 Parcelas do inventário criadas: {len(parcelas)} parcelas")

        return parcelas

    def _pontos_na_parcela(self, x_coords, y_coords, parcela_geom) -> np.ndarray:
        """Verifica quais pontos estão dentro de uma parcela"""
        try:
            # Criar pontos
            points = [Point(x, y) for x, y in zip(x_coords, y_coords)]

            # Verificar quais estão dentro do polígono
            within_polygon = np.array([
                parcela_geom['geometry'].contains(point) for point in points
            ])

            return within_polygon

        except Exception:
            # Fallback: método mais simples baseado em distância
            if hasattr(parcela_geom['geometry'], 'centroid'):
                centro = parcela_geom['geometry'].centroid
                raio = 11.28  # Raio padrão

                distan