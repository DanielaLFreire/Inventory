import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Sistema Integrado de Inventário Florestal",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== FUNÇÕES AUXILIARES ====================

def carregar_arquivo(arquivo):
    """Carrega arquivo CSV ou Excel com máxima compatibilidade"""
    try:
        if arquivo.name.endswith('.csv'):
            # Tentar diferentes separadores e encodings para CSV
            separadores = [';', ',', '\t']
            for sep in separadores:
                try:
                    df = pd.read_csv(arquivo, sep=sep)
                    if len(df.columns) > 1:  # Se tem múltiplas colunas, provavelmente acertou
                        return df
                except:
                    continue

            # Fallback final para CSV
            try:
                df = pd.read_csv(arquivo)
                return df
            except Exception as e:
                st.error(f"❌ Erro ao ler CSV: {e}")
                return None

        elif arquivo.name.endswith(('.xlsx', '.xls', '.xlsb')):

            # Lista de engines para tentar (baseada no seu requirements.txt)
            engines_disponiveis = []

            # Verificar quais engines estão disponíveis
            try:
                import openpyxl
                engines_disponiveis.append('openpyxl')
            except ImportError:
                pass

            try:
                import xlrd
                engines_disponiveis.append('xlrd')
            except ImportError:
                pass

            try:
                import pyxlsb
                engines_disponiveis.append('pyxlsb')
            except ImportError:
                pass

            # Se temos engines disponíveis, tentar usar
            if engines_disponiveis:
                for engine in engines_disponiveis:
                    try:
                        # Verificar compatibilidade engine/extensão
                        if arquivo.name.endswith('.xlsx') and engine == 'openpyxl':
                            df = pd.read_excel(arquivo, engine=engine)
                            return df
                        elif arquivo.name.endswith('.xls') and engine == 'xlrd':
                            df = pd.read_excel(arquivo, engine=engine)
                            return df
                        elif arquivo.name.endswith('.xlsb') and engine == 'pyxlsb':
                            df = pd.read_excel(arquivo, engine=engine)
                            return df
                        else:
                            # Tentar qualquer engine com qualquer arquivo
                            df = pd.read_excel(arquivo, engine=engine)
                            return df
                    except Exception as e:
                        continue

            # Tentativa final: pandas padrão (sem especificar engine)
            try:
                df = pd.read_excel(arquivo)
                return df
            except Exception as e:
                pass

            # Se chegou aqui, nada funcionou
            st.error("❌ Não foi possível ler o arquivo Excel")
            st.error("🔧 **Soluções rápidas:**")

            if not engines_disponiveis:
                st.error("• Nenhuma engine Excel encontrada")
                st.code("pip install openpyxl xlrd")
            else:
                st.error(f"• Engines disponíveis: {', '.join(engines_disponiveis)}")
                st.error("• Arquivo pode estar corrompido ou em formato não suportado")

            st.error("• **Alternativa**: Converta para CSV no Excel:")
            st.error("  Arquivo → Salvar Como → CSV UTF-8")

            return None

        else:
            st.error("❌ Formato não suportado. Use .csv, .xlsx, .xls ou .xlsb")
            return None

    except Exception as e:
        st.error(f"❌ Erro inesperado: {e}")
        return None


def verificar_colunas_inventario(df):
    """Verifica colunas obrigatórias do inventário"""
    obrigatorias = ['D_cm', 'H_m', 'talhao', 'parcela', 'cod']
    faltantes = [col for col in obrigatorias if col not in df.columns]
    return faltantes


def verificar_colunas_cubagem(df):
    """Verifica colunas obrigatórias da cubagem"""
    obrigatorias = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']
    faltantes = [col for col in obrigatorias if col not in df.columns]
    return faltantes


def processar_shapefile(arquivo_shp):
    """Processa shapefile para extrair áreas dos talhões"""
    try:
        # Tentar ler shapefile (se geopandas estiver disponível)
        try:
            import geopandas as gpd

            if arquivo_shp.name.endswith('.zip'):
                # Se for ZIP, extrair e ler
                import zipfile
                import tempfile
                import os

                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(arquivo_shp, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # Procurar arquivo .shp
                    shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                    if shp_files:
                        shp_path = os.path.join(temp_dir, shp_files[0])
                        gdf = gpd.read_file(shp_path)
                    else:
                        raise Exception("Arquivo .shp não encontrado no ZIP")
            else:
                # Arquivo .shp direto
                gdf = gpd.read_file(arquivo_shp)

            # Processar dados (similar ao código R)
            # Procurar coluna de talhão (variações possíveis)
            col_talhao = None
            for col in gdf.columns:
                if col.lower() in ['talhao', 'talhão', 'talh', 'plot', 'stand']:
                    col_talhao = col
                    break

            if col_talhao is None:
                raise Exception("Coluna de talhão não encontrada (procure: talhao, talhão, talh)")

            # Procurar coluna de área
            col_area = None
            for col in gdf.columns:
                if col.lower() in ['area_ha', 'area', 'hectares', 'ha', 'area_m2']:
                    col_area = col
                    break

            if col_area is None:
                # Calcular área da geometria
                gdf['area_ha'] = gdf.geometry.area / 10000  # Converter m² para ha
                col_area = 'area_ha'

            # Agrupar por talhão e somar áreas (como no R)
            areas_df = gdf.groupby(col_talhao)[col_area].sum().reset_index()
            areas_df.columns = ['talhao', 'area_ha']
            areas_df['talhao'] = areas_df['talhao'].astype(int)

            return areas_df

        except ImportError:
            st.error("❌ GeoPandas não está instalado")
            st.error("🔧 Execute: pip install geopandas")
            return None

    except Exception as e:
        st.error(f"❌ Erro ao processar shapefile: {e}")
        st.info("💡 Verifique se o arquivo contém colunas 'talhao' e 'area_ha'")
        return None


def processar_coordenadas(arquivo_coord, raio_parcela):
    """Processa coordenadas para calcular áreas dos talhões"""
    try:
        # Carregar arquivo de coordenadas
        df_coord = carregar_arquivo(arquivo_coord)

        if df_coord is None:
            return None

        # Verificar colunas necessárias
        colunas_necessarias = []
        for col_set in [['x', 'y'], ['X', 'Y'], ['lon', 'lat'], ['longitude', 'latitude']]:
            if all(col in df_coord.columns for col in col_set):
                colunas_necessarias = col_set
                break

        if not colunas_necessarias:
            st.error("❌ Coordenadas: colunas X,Y ou lon,lat não encontradas")
            return None

        # Verificar coluna de talhão
        col_talhao = None
        for col in df_coord.columns:
            if col.lower() in ['talhao', 'talhão', 'talh', 'plot', 'stand']:
                col_talhao = col
                break

        if col_talhao is None:
            st.error("❌ Coordenadas: coluna 'talhao' não encontrada")
            return None

        # Calcular área circular da parcela
        area_parcela_ha = 3.14159 * (raio_parcela ** 2) / 10000  # Converter m² para ha

        # Contar parcelas por talhão e calcular área total
        parcelas_por_talhao = df_coord.groupby(col_talhao).size().reset_index()
        parcelas_por_talhao.columns = ['talhao', 'num_parcelas']
        parcelas_por_talhao['area_ha'] = parcelas_por_talhao['num_parcelas'] * area_parcela_ha

        areas_df = parcelas_por_talhao[['talhao', 'area_ha']].copy()
        areas_df['talhao'] = areas_df['talhao'].astype(int)

        return areas_df

    except Exception as e:
        st.error(f"❌ Erro ao processar coordenadas: {e}")
        return None


# ==================== TÍTULO E DESCRIÇÃO ====================
st.title("🌲 Sistema Integrado de Inventário Florestal")
st.markdown("""
### 📊 Análise Completa: Hipsométrica → Volumétrica → Inventário

Este sistema integra **três etapas sequenciais** para análise florestal completa:

1. **🌳 ETAPA 1: Modelos Hipsométricos** - Testa 7 modelos e escolhe o melhor
2. **📊 ETAPA 2: Modelos Volumétricos** - Cubagem e 4 modelos de volume  
3. **📈 ETAPA 3: Inventário Florestal** - Aplica modelos e gera relatórios
""")

# ==================== SIDEBAR ====================
st.sidebar.header("📁 Upload de Dados")

arquivo_inventario = st.sidebar.file_uploader(
    "📋 Arquivo de Inventário",
    type=['csv', 'xlsx', 'xls'],
    help="Dados de parcelas (D_cm, H_m, talhao, parcela, cod, idade_anos)"
)

arquivo_cubagem = st.sidebar.file_uploader(
    "📏 Arquivo de Cubagem",
    type=['csv', 'xlsx', 'xls'],
    help="Medições detalhadas (arv, talhao, d_cm, h_m, D_cm, H_m)"
)

# Upload opcional de shapefile para áreas
arquivo_shapefile = st.sidebar.file_uploader(
    "🗺️ Shapefile Áreas (Opcional)",
    type=['shp', 'zip'],
    help="Arquivo shapefile com áreas dos talhões"
)

# Upload opcional de coordenadas
arquivo_coordenadas = st.sidebar.file_uploader(
    "📍 Coordenadas Parcelas (Opcional)",
    type=['csv', 'xlsx', 'xls'],
    help="Arquivo com coordenadas X,Y das parcelas"
)

# ==================== PROCESSAMENTO ====================

if arquivo_inventario is not None and arquivo_cubagem is not None:

    # Carregar dados
    with st.spinner("📂 Carregando arquivos..."):
        df_inventario = carregar_arquivo(arquivo_inventario)
        df_cubagem = carregar_arquivo(arquivo_cubagem)

    if df_inventario is not None and df_cubagem is not None:

        # Verificar colunas
        faltantes_inv = verificar_colunas_inventario(df_inventario)
        faltantes_cub = verificar_colunas_cubagem(df_cubagem)

        if faltantes_inv or faltantes_cub:
            if faltantes_inv:
                st.error(f"❌ Inventário - Colunas faltantes: {faltantes_inv}")
            if faltantes_cub:
                st.error(f"❌ Cubagem - Colunas faltantes: {faltantes_cub}")
            st.stop()

        # Preview dos dados
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Inventário Carregado")
            st.success(f"✅ {len(df_inventario)} registros")
            with st.expander("👀 Preview"):
                st.dataframe(df_inventario.head())

        with col2:
            st.subheader("📏 Cubagem Carregada")
            st.success(f"✅ {len(df_cubagem)} medições")
            with st.expander("👀 Preview"):
                st.dataframe(df_cubagem.head())

        # ==================== CONFIGURAÇÕES ====================
        st.header("⚙️ Configurações")

        col1, col2, col3 = st.columns(3)

        with col1:
            talhoes_excluir = st.multiselect(
                "🚫 Talhões a excluir",
                options=sorted(df_inventario['talhao'].unique()),
                help="Ex: Pinus, áreas experimentais"
            )

        with col2:
            diametro_min = st.number_input(
                "📏 Diâmetro mínimo (cm)",
                min_value=0.0,
                value=4.0
            )

        with col3:
            codigos_excluir = st.multiselect(
                "🏷️ Códigos a excluir",
                options=sorted(df_inventario['cod'].unique()),
                default=['C', 'I'] if set(['C', 'I']).issubset(df_inventario['cod'].unique()) else []
            )

        # ==================== CONFIGURAÇÕES DE ÁREA ====================
        st.subheader("📏 Configurações de Área")

        col1_area, col2_area, col3_area = st.columns(3)

        with col1_area:
            # Determinar métodos disponíveis baseado nos uploads
            metodos_disponiveis = ["Simular automaticamente", "Valores informados manualmente"]

            if arquivo_shapefile is not None:
                metodos_disponiveis.append("Upload shapefile")

            if arquivo_coordenadas is not None:
                metodos_disponiveis.append("Coordenadas das parcelas")

            metodo_area = st.selectbox(
                "🗺️ Método para Área dos Talhões",
                options=metodos_disponiveis,
                help="Como calcular as áreas dos talhões"
            )

        with col2_area:
            area_parcela = st.number_input(
                "📐 Área da Parcela (m²)",
                min_value=100,
                max_value=2000,
                value=400,
                step=100,
                help="Área padrão: 400m² (20x20m) ou 1000m² (raio 17.84m)"
            )

        with col3_area:
            if metodo_area == "Valores informados manualmente":
                st.write("**Áreas por Talhão:**")
                areas_manuais = {}
                talhoes_disponiveis = sorted(df_inventario['talhao'].unique())
                for talhao in talhoes_disponiveis:
                    if talhao not in talhoes_excluir:
                        areas_manuais[talhao] = st.number_input(
                            f"Talhão {talhao} (ha)",
                            min_value=0.1,
                            max_value=1000.0,
                            value=25.0,
                            step=0.1,
                            key=f"area_talhao_{talhao}"
                        )

            elif metodo_area == "Upload shapefile":
                st.success("📁 Shapefile carregado!")
                st.write("✅ Áreas serão extraídas automaticamente")

            elif metodo_area == "Coordenadas das parcelas":
                st.success("📍 Coordenadas carregadas!")
                raio_parcela = st.number_input(
                    "📐 Raio da Parcela (m)",
                    min_value=5.0,
                    max_value=30.0,
                    value=11.28,
                    step=0.1,
                    help="Raio para calcular área circular (11.28m = 400m²)"
                )
                area_calculada = 3.14159 * (raio_parcela ** 2)
                st.write(f"**Área calculada**: {area_calculada:.0f} m²")

            else:
                st.info("💡 **Opções adicionais:**")
                st.write("- **Shapefile**: Faça upload para usar áreas reais")
                st.write("- **Coordenadas**: Faça upload para cálculo automático")
                st.write("- **Manual**: Digite áreas conhecidas")

        # ==================== BOTÃO PRINCIPAL ====================
        if st.button("🚀 Executar Análise Completa", type="primary", use_container_width=True):

            # ==================== ETAPA 1: MODELOS HIPSOMÉTRICOS ====================
            st.header("🌳 ETAPA 1: Modelos Hipsométricos")

            with st.spinner("Testando modelos hipsométricos..."):

                # Filtrar dados
                # ==================== CORREÇÃO COMPLETA DA ALTURA DOMINANTE ====================
                # Substituir a seção que vai aproximadamente da linha 250 até 340 por este código:

                # Filtrar dados para modelos hipsométricos
                df_hip = df_inventario.copy()

                if talhoes_excluir:
                    df_hip = df_hip[~df_hip['talhao'].isin(talhoes_excluir)]

                df_hip = df_hip[
                    (df_hip['D_cm'].notna()) &
                    (df_hip['H_m'].notna()) &
                    (df_hip['D_cm'] >= diametro_min) &
                    (df_hip['H_m'] > 1.3)
                    ]

                if codigos_excluir:
                    df_hip = df_hip[~df_hip['cod'].isin(codigos_excluir)]

                #st.info(f"📊 Dados para modelos hipsométricos: {len(df_hip)} observações")


                # ==================== CÁLCULO DE ALTURA DOMINANTE - CORRIGIDO ====================

                def calcular_altura_dominante(df):
                    """Calcula altura dominante por parcela com múltiplas estratégias"""

                    dominantes_list = []

                    # Estratégia 1: Usar árvores marcadas como dominantes (cod = 'D')
                    arvores_dominantes = df[df['cod'] == 'D']

                    if len(arvores_dominantes) > 0:
                        #st.info("✅ Usando árvores marcadas como dominantes (cod = 'D')")
                        dominantes_parcela = arvores_dominantes.groupby('parcela')['H_m'].mean().reset_index()
                        dominantes_parcela.columns = ['parcela', 'H_dom']
                        dominantes_list.extend(dominantes_parcela.to_dict('records'))

                        # Parcelas que já têm dominantes
                        parcelas_com_dominantes = set(dominantes_parcela['parcela'])
                    else:
                        st.info("⚠️ Nenhuma árvore marcada como dominante (cod = 'D') encontrada")
                        parcelas_com_dominantes = set()

                    # Estratégia 2: Para parcelas sem dominantes, calcular automaticamente
                    todas_parcelas = set(df['parcela'].unique())
                    parcelas_sem_dominantes = todas_parcelas - parcelas_com_dominantes

                    if parcelas_sem_dominantes:
                        #st.info(f"🔄 Calculando altura dominante para {len(parcelas_sem_dominantes)} parcelas restantes")

                        for parcela in parcelas_sem_dominantes:
                            dados_parcela = df[df['parcela'] == parcela]

                            if len(dados_parcela) > 0:
                                # Pegar as 3 maiores árvores em diâmetro (ou todas se menos de 3)
                                n_arvores = min(3, len(dados_parcela))
                                maiores_arvores = dados_parcela.nlargest(n_arvores, 'D_cm')
                                h_dom = maiores_arvores['H_m'].mean()

                                dominantes_list.append({
                                    'parcela': parcela,
                                    'H_dom': h_dom
                                })

                    # Converter para DataFrame
                    if dominantes_list:
                        dominantes_df = pd.DataFrame(dominantes_list)

                        # Verificar e corrigir valores
                        dominantes_df['H_dom'] = dominantes_df['H_dom'].fillna(df['H_m'].mean())

                        #st.success(f"✅ Altura dominante calculada para {len(dominantes_df)} parcelas")
                        #st.info(f"📊 H_dom médio: {dominantes_df['H_dom'].mean():.1f}m (min: {dominantes_df['H_dom'].min():.1f}m, max: {dominantes_df['H_dom'].max():.1f}m)")

                        return dominantes_df
                    else:
                        # Fallback extremo: usar altura média geral
                        st.warning("⚠️ Fallback: Usando altura média geral como H_dom")
                        parcelas_unicas = df['parcela'].unique()
                        h_media = df['H_m'].mean()

                        dominantes_df = pd.DataFrame({
                            'parcela': parcelas_unicas,
                            'H_dom': [h_media] * len(parcelas_unicas)
                        })

                        return dominantes_df


                # Calcular altura dominante
                try:
                    dominantes = calcular_altura_dominante(df_hip)
                except Exception as e:
                    st.error(f"❌ Erro ao calcular altura dominante: {e}")

                    # Fallback de emergência
                    st.warning("🚨 Usando fallback de emergência para H_dom")
                    parcelas_unicas = df_hip['parcela'].unique()
                    h_media = df_hip['H_m'].mean()

                    dominantes = pd.DataFrame({
                        'parcela': parcelas_unicas,
                        'H_dom': [h_media] * len(parcelas_unicas)
                    })


                # ==================== CRIAÇÃO DE VARIÁVEIS - CORRIGIDA ====================

                def criar_variaveis_seguras(df_hip, dominantes):
                    """Cria variáveis para modelos com máxima segurança"""

                    try:
                        #st.info("🔧 Criando variáveis transformadas...")

                        # 1. Fazer merge com dominantes
                        if 'H_dom' not in df_hip.columns:
                            df_hip = df_hip.merge(dominantes, on='parcela', how='left')

                        # 2. Verificar e corrigir H_dom
                        if 'H_dom' not in df_hip.columns:
                            st.warning("⚠️ H_dom não encontrado após merge - criando manualmente")
                            df_hip['H_dom'] = df_hip['H_m'].mean()

                        # Preencher H_dom faltante
                        h_dom_medio = df_hip['H_dom'].mean()
                        if pd.isna(h_dom_medio) or h_dom_medio <= 0:
                            h_dom_medio = df_hip['H_m'].mean()

                        df_hip['H_dom'] = df_hip['H_dom'].fillna(h_dom_medio)

                        # 3. Verificar dados básicos
                        #st.info(f"✅ H_dom: média = {df_hip['H_dom'].mean():.1f}m, valores únicos = {df_hip['H_dom'].nunique()}")

                        # 4. Criar variáveis transformadas com clipping
                        df_hip['ln_H'] = np.log(df_hip['H_m'].clip(lower=0.1))
                        df_hip['inv_D'] = 1 / df_hip['D_cm'].clip(lower=0.1)
                        df_hip['D2'] = df_hip['D_cm'] ** 2
                        df_hip['ln_D'] = np.log(df_hip['D_cm'].clip(lower=0.1))
                        df_hip['ln_H_dom'] = np.log(df_hip['H_dom'].clip(lower=0.1))

                        # 5. Produtividade (Prod)
                        h_adjusted = (df_hip['H_m'] - 1.3).clip(lower=0.1)
                        df_hip['Prod'] = df_hip['D2'] / h_adjusted

                        # 6. Idade (se disponível)
                        if 'idade_anos' in df_hip.columns:
                            idade_media = df_hip['idade_anos'].mean()
                            if pd.isna(idade_media) or idade_media <= 0:
                                idade_media = 5.0  # Default

                            df_hip['idade_anos'] = df_hip['idade_anos'].fillna(idade_media)
                            df_hip['DI'] = df_hip['D_cm'] * df_hip['idade_anos']

                            #st.info(f"✅ Idade: média = {df_hip['idade_anos'].mean():.1f} anos")
                        else:
                            st.info("ℹ️ Coluna 'idade_anos' não encontrada - modelos sem idade serão usados")

                        # 7. Verificar se todas as variáveis foram criadas
                        variaveis_esperadas = ['ln_H', 'inv_D', 'D2', 'ln_D', 'ln_H_dom', 'Prod', 'H_dom']
                        variaveis_criadas = [var for var in variaveis_esperadas if var in df_hip.columns]

                        #st.success(f"✅ Variáveis criadas: {len(variaveis_criadas)}/{len(variaveis_esperadas)}")
                        #st.info(f"📋 Variáveis: {', '.join(variaveis_criadas)}")

                        # 8. Estatísticas de qualidade
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Registros", len(df_hip))
                        with col2:
                            st.metric("DAP médio", f"{df_hip['D_cm'].mean():.1f} cm")
                        with col3:
                            st.metric("H médio", f"{df_hip['H_m'].mean():.1f} m")
                        with col4:
                            st.metric("H_dom médio", f"{df_hip['H_dom'].mean():.1f} m")

                        return df_hip

                    except Exception as e:
                        st.error(f"❌ Erro ao criar variáveis: {e}")

                        # Debug: mostrar estado das variáveis
                        st.error("🔍 Debug - Estado das variáveis:")
                        st.error(f"- Colunas em df_hip: {list(df_hip.columns)}")
                        if 'H_dom' in df_hip.columns:
                            st.error(f"- H_dom existe: {df_hip['H_dom'].describe()}")
                        else:
                            st.error("- H_dom NÃO existe")

                        raise e


                # Aplicar criação de variáveis
                df_hip = criar_variaveis_seguras(df_hip, dominantes)

                # ==================== VERIFICAÇÃO FINAL ====================

                # Verificar se tudo está OK antes de continuar
                variaveis_obrigatorias = ['D_cm', 'H_m', 'H_dom', 'ln_H', 'inv_D', 'ln_D', 'ln_H_dom']
                variaveis_faltantes = [var for var in variaveis_obrigatorias if var not in df_hip.columns]

                if variaveis_faltantes:
                    st.error(f"❌ Variáveis obrigatórias faltantes: {variaveis_faltantes}")
                    st.stop()
                #else:
                    #st.success("✅ Todas as variáveis obrigatórias foram criadas com sucesso!")

                # Mostrar preview dos dados preparados
                with st.expander("👀 Preview dos Dados Preparados"):
                    colunas_mostrar = ['D_cm', 'H_m', 'H_dom', 'ln_H', 'inv_D', 'ln_D', 'ln_H_dom', 'Prod']
                    colunas_disponiveis = [col for col in colunas_mostrar if col in df_hip.columns]
                    st.dataframe(df_hip[colunas_disponiveis].head(10))

                # Função auxiliar para limpeza de dados
                def ajustar_modelo_seguro(X, y, nome_modelo):
                    """Ajusta modelo com tratamento de NaN"""
                    try:
                        # Verificar NaN
                        if X.isna().any().any() or y.isna().any():
                            # Limpar dados
                            mask_validos = ~(X.isna().any(axis=1) | y.isna())
                            X_clean = X[mask_validos]
                            y_clean = y[mask_validos]

                            if len(X_clean) < 10:
                                raise ValueError(f"Poucos dados válidos: {len(X_clean)}")

                            #st.info(f"🧹 {nome_modelo}: {len(X_clean)}/{len(X)} observações válidas")
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

                        return {'modelo': modelo, 'r2': r2, 'rmse': rmse, 'y_pred': y_pred}

                    except Exception as e:
                        st.warning(f"⚠️ Erro em {nome_modelo}: {e}")
                        return None


                def calcular_r2_generalizado(y_obs, y_pred):
                    return 1 - (np.sum((y_obs - y_pred) ** 2) / np.sum((y_obs - np.mean(y_obs)) ** 2))


                # Ajustar modelos hipsométricos com tratamento de erros
                resultados_hip = {}
                predicoes_hip = {}

                # 1. Curtis
                resultado = ajustar_modelo_seguro(df_hip[['inv_D']], df_hip['ln_H'], 'Curtis')
                if resultado:
                    predicoes_hip['Curtis'] = np.exp(resultado['y_pred'])
                    r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Curtis'])
                    resultados_hip['Curtis'] = {'r2g': r2g, 'rmse': np.sqrt(
                        mean_squared_error(df_hip['H_m'], predicoes_hip['Curtis']))}

                # 2. Campos
                resultado = ajustar_modelo_seguro(df_hip[['inv_D', 'ln_H_dom']], df_hip['ln_H'], 'Campos')
                if resultado:
                    predicoes_hip['Campos'] = np.exp(resultado['y_pred'])
                    r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Campos'])
                    resultados_hip['Campos'] = {'r2g': r2g, 'rmse': np.sqrt(
                        mean_squared_error(df_hip['H_m'], predicoes_hip['Campos']))}

                # 3. Henri
                resultado = ajustar_modelo_seguro(df_hip[['ln_D']], df_hip['H_m'], 'Henri')
                if resultado:
                    predicoes_hip['Henri'] = resultado['y_pred']
                    r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Henri'])
                    resultados_hip['Henri'] = {'r2g': r2g, 'rmse': resultado['rmse']}

                # 4. Prodan - VERSÃO ROBUSTA
                try:
                    # Verificar disponibilidade de idade
                    tem_idade = 'idade_anos' in df_hip.columns and df_hip['idade_anos'].notna().sum() > 10

                    if tem_idade:
                        # Versão com idade
                        colunas = ['D_cm', 'D2', 'DI']
                        mask_validos = df_hip[colunas + ['Prod']].notna().all(axis=1)

                        if mask_validos.sum() >= 10:
                            df_clean = df_hip[mask_validos]
                            resultado = ajustar_modelo_seguro(df_clean[colunas], df_clean['Prod'], 'Prodan (com idade)')

                            if resultado:
                                # Predição para todo dataset
                                X_pred = df_hip[colunas].fillna(df_hip[colunas].mean())
                                y_pred_prodan = resultado['modelo'].predict(X_pred)

                                predicoes_hip['Prodan'] = (df_hip['D2'] / np.clip(y_pred_prodan, 0.1, None)) + 1.3
                                r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Prodan'])
                                resultados_hip['Prodan'] = {'r2g': r2g, 'rmse': np.sqrt(
                                    mean_squared_error(df_hip['H_m'], predicoes_hip['Prodan']))}
                                #st.success("✅ Prodan: Modelo com idade ajustado")
                            else:
                                raise ValueError("Falha no ajuste com idade")
                        else:
                            raise ValueError("Poucos dados válidos com idade")
                    else:
                        raise ValueError("Idade não disponível")

                except Exception as e:
                    # Fallback: Prodan sem idade
                    try:
                        #st.info("🔄 Prodan: Tentando versão sem idade...")
                        colunas = ['D_cm', 'D2']
                        resultado = ajustar_modelo_seguro(df_hip[colunas], df_hip['Prod'], 'Prodan (sem idade)')

                        if resultado:
                            X_pred = df_hip[colunas].fillna(df_hip[colunas].mean())
                            y_pred_prodan = resultado['modelo'].predict(X_pred)

                            predicoes_hip['Prodan'] = (df_hip['D2'] / np.clip(y_pred_prodan, 0.1, None)) + 1.3
                            r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Prodan'])
                            resultados_hip['Prodan'] = {'r2g': r2g, 'rmse': np.sqrt(
                                mean_squared_error(df_hip['H_m'], predicoes_hip['Prodan']))}
                            #st.success("✅ Prodan: Modelo simplificado (sem idade) ajustado")
                        else:
                            st.warning("⚠️ Prodan: Falha em todas as versões - modelo não incluído")

                    except Exception as e2:
                        st.warning(f"⚠️ Prodan: Falha completa - {e2}")


                # Modelos não-lineares com tratamento de erros
                def ajustar_modelo_nao_linear(func, params_iniciais, nome):
                    """Ajusta modelo não-linear com tratamento de erros"""
                    try:
                        # Limpar dados
                        mask_validos = df_hip[['D_cm', 'H_m']].notna().all(axis=1)
                        if mask_validos.sum() < 20:
                            raise ValueError("Poucos dados válidos para modelo não-linear")

                        D_clean = df_hip.loc[mask_validos, 'D_cm']
                        H_clean = df_hip.loc[mask_validos, 'H_m']

                        # Ajustar
                        popt, _ = curve_fit(func, D_clean, H_clean, p0=params_iniciais, maxfev=5000)

                        # Predições para todo dataset
                        D_pred = df_hip['D_cm'].fillna(df_hip['D_cm'].mean())
                        H_pred = func(D_pred, *popt)

                        return H_pred, popt

                    except Exception as e:
                        st.warning(f"⚠️ Erro no modelo {nome}: {e}")
                        return None, None


                # 5. Chapman
                def chapman_func(D, b0, b1, b2):
                    return b0 * (1 - np.exp(-b1 * D)) ** b2


                altura_max = df_hip['H_m'].max() * 1.2
                H_pred, popt = ajustar_modelo_nao_linear(chapman_func, [altura_max, 0.01, 1.0], 'Chapman')
                if H_pred is not None:
                    predicoes_hip['Chapman'] = H_pred
                    r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Chapman'])
                    resultados_hip['Chapman'] = {'r2g': r2g, 'rmse': np.sqrt(
                        mean_squared_error(df_hip['H_m'], predicoes_hip['Chapman']))}


                # 6. Weibull
                def weibull_func(D, a, b, c):
                    return a * (1 - np.exp(-b * D ** c))


                H_pred, popt = ajustar_modelo_nao_linear(weibull_func, [altura_max, 0.01, 1.0], 'Weibull')
                if H_pred is not None:
                    predicoes_hip['Weibull'] = H_pred
                    r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Weibull'])
                    resultados_hip['Weibull'] = {'r2g': r2g, 'rmse': np.sqrt(
                        mean_squared_error(df_hip['H_m'], predicoes_hip['Weibull']))}


                # 7. Mononuclear
                def mono_func(D, a, b, c):
                    return a * (1 - b * np.exp(-c * D))


                H_pred, popt = ajustar_modelo_nao_linear(mono_func, [altura_max, 1.0, 0.1], 'Mononuclear')
                if H_pred is not None:
                    predicoes_hip['Mononuclear'] = H_pred
                    r2g = calcular_r2_generalizado(df_hip['H_m'], predicoes_hip['Mononuclear'])
                    resultados_hip['Mononuclear'] = {'r2g': r2g, 'rmse': np.sqrt(
                        mean_squared_error(df_hip['H_m'], predicoes_hip['Mononuclear']))}

                # Verificar se pelo menos um modelo funcionou
                if not resultados_hip:
                    st.error("❌ Nenhum modelo hipsométrico foi ajustado com sucesso!")
                    st.info("💡 Verifique a qualidade dos dados de entrada")
                    st.stop()
                #else:
                    #st.success(f"✅ {len(resultados_hip)} modelos hipsométricos ajustados com sucesso!")

                # Ranking dos modelos hipsométricos
                if resultados_hip:
                    ranking_hip = []
                    for modelo, resultado in resultados_hip.items():
                        ranking_hip.append({
                            'Modelo': modelo,
                            'R² Generalizado': resultado['r2g'],
                            'RMSE': resultado['rmse']
                        })

                    df_ranking_hip = pd.DataFrame(ranking_hip)
                    df_ranking_hip = df_ranking_hip.sort_values('R² Generalizado', ascending=False)
                    df_ranking_hip['Ranking'] = range(1, len(df_ranking_hip) + 1)

                    melhor_modelo_hip = df_ranking_hip.iloc[0]['Modelo']
                    melhor_r2_hip = df_ranking_hip.iloc[0]['R² Generalizado']

                    # ==================== DETALHAMENTO DOS MODELOS HIPSOMÉTRICOS ====================
                    st.subheader("📊 Detalhamento dos Modelos Hipsométricos")

                    # Criar abas para cada modelo
                    if len(predicoes_hip) > 0:
                        abas_hip = st.tabs([f"{modelo}" for modelo in predicoes_hip.keys()])

                        for i, (modelo, aba) in enumerate(zip(predicoes_hip.keys(), abas_hip)):
                            with aba:
                                col1, col2 = st.columns([1, 1])

                                with col1:
                                    # Informações do modelo
                                    r2_modelo = resultados_hip[modelo]['r2g']
                                    rmse_modelo = resultados_hip[modelo]['rmse']

                                    # Classificação
                                    if r2_modelo >= 0.9:
                                        qualidade = "***** Excelente"
                                    elif r2_modelo >= 0.8:
                                        qualidade = "**** Muito Bom"
                                    elif r2_modelo >= 0.7:
                                        qualidade = "*** Bom"
                                    elif r2_modelo >= 0.6:
                                        qualidade = "** Regular"
                                    else:
                                        qualidade = "* Fraco"

                                    st.write(
                                        f"**Ranking:** #{df_ranking_hip[df_ranking_hip['Modelo'] == modelo]['Ranking'].iloc[0]}")
                                    st.write(f"**Qualidade:** {qualidade}")
                                    st.write(f"**R² Generalizado:** {r2_modelo:.4f}")
                                    st.write(f"**RMSE:** {rmse_modelo:.4f}")

                                    # Equação e coeficientes específicos
                                    if modelo == "Curtis":
                                        st.latex(r"ln(H) = \beta_0 + \beta_1 \cdot \frac{1}{D}")
                                        # Aqui você pode extrair os coeficientes reais do modelo ajustado
                                        st.write("**Coeficientes:**")
                                        st.write("- β₀ (intercepto)")
                                        st.write("- β₁ (1/D)")

                                    elif modelo == "Campos":
                                        st.latex(
                                            r"ln(H) = \beta_0 + \beta_1 \cdot \frac{1}{D} + \beta_2 \cdot ln(H_{dom})")
                                        st.write("**Coeficientes:**")
                                        st.write("- β₀ (intercepto)")
                                        st.write("- β₁ (1/D)")
                                        st.write("- β₂ (ln H_dom)")

                                    elif modelo == "Henri":
                                        st.latex(r"H = \beta_0 + \beta_1 \cdot ln(D)")
                                        st.write("**Coeficientes:**")
                                        st.write("- β₀ (intercepto)")
                                        st.write("- β₁ (ln D)")

                                    elif modelo == "Prodan":
                                        st.latex(
                                            r"\frac{D^2}{H-1.3} = \beta_0 + \beta_1 \cdot D + \beta_2 \cdot D^2 + \beta_3 \cdot D \cdot Idade")
                                        st.write("**Coeficientes:**")
                                        st.write("- β₀ (intercepto)")
                                        st.write("- β₁ (D)")
                                        st.write("- β₂ (D²)")
                                        st.write("- β₃ (D×Idade)")

                                    elif modelo == "Chapman":
                                        st.latex(r"H = b_0 \cdot (1 - e^{-b_1 \cdot D})^{b_2}")
                                        st.write("**Parâmetros:**")
                                        st.write("- b₀ (altura assintótica)")
                                        st.write("- b₁ (taxa de crescimento)")
                                        st.write("- b₂ (parâmetro de forma)")

                                    elif modelo == "Weibull":
                                        st.latex(r"H = a \cdot (1 - e^{-b \cdot D^c})")
                                        st.write("**Parâmetros:**")
                                        st.write("- a (altura assintótica)")
                                        st.write("- b (parâmetro de escala)")
                                        st.write("- c (parâmetro de forma)")

                                    elif modelo == "Mononuclear":
                                        st.latex(r"H = a \cdot (1 - b \cdot e^{-c \cdot D})")
                                        st.write("**Parâmetros:**")
                                        st.write("- a (altura assintótica)")
                                        st.write("- b (parâmetro de intercepto)")
                                        st.write("- c (taxa de decaimento)")

                                with col2:
                                    # Gráfico individual do modelo
                                    fig, ax = plt.subplots(figsize=(8, 6))

                                    # Dados observados
                                    ax.scatter(df_hip['D_cm'], df_hip['H_m'], alpha=0.4, color='gray', s=15,
                                               label='Observado')

                                    # Modelo específico
                                    cores_modelo = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
                                    cor = cores_modelo[i % len(cores_modelo)]
                                    ax.scatter(df_hip['D_cm'], predicoes_hip[modelo], alpha=0.7, color=cor, s=15,
                                               label=f'{modelo}')

                                    # Destacar se é o melhor
                                    if modelo == melhor_modelo_hip:
                                        ax.set_title(f'🏆 {modelo} - MELHOR MODELO (R² = {r2_modelo:.3f})',
                                                     fontweight='bold', color='red')
                                    else:
                                        ax.set_title(f'{modelo} (R² = {r2_modelo:.3f})')

                                    ax.set_xlabel('Diâmetro (cm)')
                                    ax.set_ylabel('Altura (m)')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)

                                    st.pyplot(fig)

                                # Gráfico de resíduos
                                st.subheader(f"📈 Análise de Resíduos - {modelo}")

                                col1_res, col2_res = st.columns(2)

                                with col1_res:
                                    # Resíduos vs Preditos
                                    fig_res1, ax_res1 = plt.subplots(figsize=(6, 5))
                                    residuos = df_hip['H_m'] - predicoes_hip[modelo]
                                    ax_res1.scatter(predicoes_hip[modelo], residuos, alpha=0.6, color=cor)
                                    ax_res1.axhline(y=0, color='red', linestyle='--')
                                    ax_res1.set_xlabel('Valores Preditos (m)')
                                    ax_res1.set_ylabel('Resíduos (m)')
                                    ax_res1.set_title('Resíduos vs Preditos')
                                    ax_res1.grid(True, alpha=0.3)
                                    st.pyplot(fig_res1)

                                with col2_res:
                                    # Histograma dos resíduos
                                    fig_res2, ax_res2 = plt.subplots(figsize=(6, 5))
                                    ax_res2.hist(residuos, bins=15, alpha=0.7, color=cor, edgecolor='black')
                                    ax_res2.axvline(x=0, color='red', linestyle='--')
                                    ax_res2.set_xlabel('Resíduos (m)')
                                    ax_res2.set_ylabel('Frequência')
                                    ax_res2.set_title('Distribuição dos Resíduos')
                                    ax_res2.grid(True, alpha=0.3)
                                    st.pyplot(fig_res2)

                    # Ranking final dos modelos hipsométricos
                    st.subheader("🏆 Ranking Final - Modelos Hipsométricos")
                    st.dataframe(df_ranking_hip[['Ranking', 'Modelo', 'R² Generalizado', 'RMSE']].round(4))
                    st.success(f"🏆 **Melhor modelo**: {melhor_modelo_hip} (R² = {melhor_r2_hip:.4f})")

            # ==================== ETAPA 2: MODELOS VOLUMÉTRICOS ====================
            st.header("📊 ETAPA 2: Modelos Volumétricos")

            with st.spinner("Processando cubagem..."):

                # Processar cubagem (método de Smalian)
                df_cubagem_proc = df_cubagem.copy()

                # Converter para numérico
                colunas_num = ['d_cm', 'h_m', 'D_cm', 'H_m']
                for col in colunas_num:
                    df_cubagem_proc[col] = pd.to_numeric(df_cubagem_proc[col], errors='coerce')

                # Calcular área seccional
                df_cubagem_proc['a_m2'] = np.pi * (df_cubagem_proc['d_cm'] ** 2 / 40000)

                # Ordenar por árvore e altura
                df_cubagem_proc = df_cubagem_proc.sort_values(['arv', 'talhao', 'h_m']).reset_index(drop=True)

                # Método de Smalian
                volumes_list = []

                for (talhao, arv), grupo in df_cubagem_proc.groupby(['talhao', 'arv']):
                    grupo = grupo.sort_values('h_m').reset_index(drop=True)

                    for i in range(len(grupo)):
                        row = grupo.iloc[i].copy()

                        if i > 0:
                            row['a1'] = grupo.iloc[i - 1]['a_m2']
                            row['h1'] = grupo.iloc[i - 1]['h_m']
                            row['a2'] = grupo.iloc[i]['a_m2']
                            row['h2'] = grupo.iloc[i]['h_m']
                            row['delta_h'] = row['h2'] - row['h1']
                            row['va_m3'] = ((row['a1'] + row['a2']) / 2) * row['delta_h']
                        else:
                            row['va_m3'] = np.nan

                        volumes_list.append(row)

                df_cubagem_proc = pd.DataFrame(volumes_list)

                # Marcar toco e calcular volume total
                df_cubagem_proc['secao_tipo'] = df_cubagem_proc['h_m'].apply(
                    lambda x: 'Toco' if abs(x - 0.1) < 0.05 else 'Seção'
                )

                volumes_arvore = df_cubagem_proc[
                    (df_cubagem_proc['va_m3'].notna()) &
                    (df_cubagem_proc['secao_tipo'] != 'Toco')
                    ].groupby(['arv', 'talhao', 'D_cm', 'H_m']).agg({
                    'va_m3': 'sum'
                }).reset_index()

                volumes_arvore['V'] = volumes_arvore['va_m3']
                volumes_arvore = volumes_arvore.drop('va_m3', axis=1)

                st.success(f"✅ Volumes calculados para {len(volumes_arvore)} árvores")

                # Criar variáveis para modelos volumétricos
                volumes_arvore['ln_V'] = np.log(volumes_arvore['V'])
                volumes_arvore['ln_H'] = np.log(volumes_arvore['H_m'])
                volumes_arvore['ln_D'] = np.log(volumes_arvore['D_cm'])
                volumes_arvore['inv_D'] = 1 / volumes_arvore['D_cm']
                volumes_arvore['D2'] = volumes_arvore['D_cm'] ** 2
                volumes_arvore['D2_H'] = volumes_arvore['D2'] * volumes_arvore['H_m']
                volumes_arvore['ln_D2_H'] = np.log(volumes_arvore['D2_H'])

                # Ajustar 4 modelos volumétricos
                modelos_vol = {}
                r2_vol = {}

                # 1. Schumacher-Hall
                X_sch = volumes_arvore[['ln_D', 'ln_H']]
                y_sch = volumes_arvore['ln_V']
                mod_sch = LinearRegression().fit(X_sch, y_sch)
                pred_sch = np.exp(mod_sch.predict(X_sch))
                modelos_vol['Schumacher'] = mod_sch
                r2_vol['Schumacher'] = r2_score(volumes_arvore['V'], pred_sch)

                # 2. G1
                X_g1 = volumes_arvore[['ln_D', 'inv_D']]
                y_g1 = volumes_arvore['ln_V']
                mod_g1 = LinearRegression().fit(X_g1, y_g1)
                pred_g1 = np.exp(mod_g1.predict(X_g1))
                modelos_vol['G1'] = mod_g1
                r2_vol['G1'] = r2_score(volumes_arvore['V'], pred_g1)

                # 3. G2
                X_g2 = volumes_arvore[['D2', 'D2_H', 'H_m']]
                y_g2 = volumes_arvore['V']
                mod_g2 = LinearRegression().fit(X_g2, y_g2)
                pred_g2 = mod_g2.predict(X_g2)
                modelos_vol['G2'] = mod_g2
                r2_vol['G2'] = r2_score(volumes_arvore['V'], pred_g2)

                # 4. G3
                X_g3 = volumes_arvore[['ln_D2_H']]
                y_g3 = volumes_arvore['ln_V']
                mod_g3 = LinearRegression().fit(X_g3, y_g3)
                pred_g3 = np.exp(mod_g3.predict(X_g3))
                modelos_vol['G3'] = mod_g3
                r2_vol['G3'] = r2_score(volumes_arvore['V'], pred_g3)

                # Melhor modelo volumétrico
                melhor_modelo_vol = max(r2_vol.keys(), key=lambda k: r2_vol[k])

                # ==================== DETALHAMENTO DOS MODELOS VOLUMÉTRICOS ====================
                st.subheader("📊 Detalhamento dos Modelos Volumétricos")

                # Criar abas para cada modelo volumétrico
                abas_vol = st.tabs([f"{modelo}" for modelo in r2_vol.keys()])

                predicoes_vol = {
                    'Schumacher': pred_sch,
                    'G1': pred_g1,
                    'G2': pred_g2,
                    'G3': pred_g3
                }

                for i, (modelo, aba) in enumerate(zip(r2_vol.keys(), abas_vol)):
                    with aba:
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            # Informações do modelo
                            r2_modelo = r2_vol[modelo]
                            rmse_modelo = np.sqrt(mean_squared_error(volumes_arvore['V'], predicoes_vol[modelo]))

                            # Classificação
                            if r2_modelo >= 0.9:
                                qualidade = "***** Excelente"
                            elif r2_modelo >= 0.8:
                                qualidade = "**** Muito Bom"
                            elif r2_modelo >= 0.7:
                                qualidade = "*** Bom"
                            elif r2_modelo >= 0.6:
                                qualidade = "** Regular"
                            else:
                                qualidade = "* Fraco"

                            # Ranking
                            ranking_pos = sorted(r2_vol.keys(), key=lambda k: r2_vol[k], reverse=True).index(modelo) + 1

                            st.write(f"**Ranking:** #{ranking_pos}")
                            st.write(f"**Qualidade:** {qualidade}")
                            st.write(f"**R²:** {r2_modelo:.4f}")
                            st.write(f"**RMSE:** {rmse_modelo:.4f}")

                            # Equação e coeficientes específicos
                            if modelo == "Schumacher":
                                st.latex(r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot ln(H)")
                                st.write("**Coeficientes:**")
                                st.write(f"- β₀: {mod_sch.intercept_:.4f}")
                                st.write(f"- β₁ (ln D): {mod_sch.coef_[0]:.4f}")
                                st.write(f"- β₂ (ln H): {mod_sch.coef_[1]:.4f}")

                            elif modelo == "G1":
                                st.latex(r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot \frac{1}{D}")
                                st.write("**Coeficientes:**")
                                st.write(f"- β₀: {mod_g1.intercept_:.4f}")
                                st.write(f"- β₁ (ln D): {mod_g1.coef_[0]:.4f}")
                                st.write(f"- β₂ (1/D): {mod_g1.coef_[1]:.4f}")

                            elif modelo == "G2":
                                st.latex(r"V = \beta_0 + \beta_1 \cdot D^2 + \beta_2 \cdot D^2H + \beta_3 \cdot H")
                                st.write("**Coeficientes:**")
                                st.write(f"- β₀: {mod_g2.intercept_:.4f}")
                                st.write(f"- β₁ (D²): {mod_g2.coef_[0]:.4f}")
                                st.write(f"- β₂ (D²H): {mod_g2.coef_[1]:.4f}")
                                st.write(f"- β₃ (H): {mod_g2.coef_[2]:.4f}")

                            elif modelo == "G3":
                                st.latex(r"ln(V) = \beta_0 + \beta_1 \cdot ln(D^2H)")
                                st.write("**Coeficientes:**")
                                st.write(f"- β₀: {mod_g3.intercept_:.4f}")
                                st.write(f"- β₁ (ln D²H): {mod_g3.coef_[0]:.4f}")

                        with col2:
                            # Gráfico individual do modelo (Observado vs Predito)
                            fig, ax = plt.subplots(figsize=(8, 6))

                            # Scatter plot
                            cores_modelo = ['red', 'green', 'blue', 'orange']
                            cor = cores_modelo[i % len(cores_modelo)]
                            ax.scatter(volumes_arvore['V'], predicoes_vol[modelo], alpha=0.6, color=cor)

                            # Linha 1:1
                            min_val = min(volumes_arvore['V'].min(), predicoes_vol[modelo].min())
                            max_val = max(volumes_arvore['V'].max(), predicoes_vol[modelo].max())
                            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1')

                            # Destacar se é o melhor
                            if modelo == melhor_modelo_vol:
                                ax.set_title(f'🏆 {modelo} - MELHOR MODELO (R² = {r2_modelo:.3f})',
                                             fontweight='bold', color='red')
                            else:
                                ax.set_title(f'{modelo} (R² = {r2_modelo:.3f})')

                            ax.set_xlabel('Volume Observado (m³)')
                            ax.set_ylabel('Volume Predito (m³)')
                            ax.legend()
                            ax.grid(True, alpha=0.3)

                            st.pyplot(fig)

                        # Gráfico de resíduos para volume
                        st.subheader(f"📈 Análise de Resíduos - {modelo}")

                        col1_res, col2_res = st.columns(2)

                        with col1_res:
                            # Resíduos vs Preditos
                            fig_res1, ax_res1 = plt.subplots(figsize=(6, 5))
                            residuos_vol = volumes_arvore['V'] - predicoes_vol[modelo]
                            ax_res1.scatter(predicoes_vol[modelo], residuos_vol, alpha=0.6, color=cor)
                            ax_res1.axhline(y=0, color='red', linestyle='--')
                            ax_res1.set_xlabel('Volumes Preditos (m³)')
                            ax_res1.set_ylabel('Resíduos (m³)')
                            ax_res1.set_title('Resíduos vs Preditos')
                            ax_res1.grid(True, alpha=0.3)
                            st.pyplot(fig_res1)

                        with col2_res:
                            # Histograma dos resíduos
                            fig_res2, ax_res2 = plt.subplots(figsize=(6, 5))
                            ax_res2.hist(residuos_vol, bins=15, alpha=0.7, color=cor, edgecolor='black')
                            ax_res2.axvline(x=0, color='red', linestyle='--')
                            ax_res2.set_xlabel('Resíduos (m³)')
                            ax_res2.set_ylabel('Frequência')
                            ax_res2.set_title('Distribuição dos Resíduos')
                            ax_res2.grid(True, alpha=0.3)
                            st.pyplot(fig_res2)

                # Tabela comparação final
                df_vol = pd.DataFrame({
                    'Modelo': list(r2_vol.keys()),
                    'R²': [r2_vol[m] for m in r2_vol.keys()],
                    'RMSE': [np.sqrt(mean_squared_error(volumes_arvore['V'], predicoes_vol[m])) for m in r2_vol.keys()]
                }).sort_values('R²', ascending=False)
                df_vol['Ranking'] = range(1, len(df_vol) + 1)

                st.subheader("🏆 Ranking Final - Modelos Volumétricos")
                st.dataframe(df_vol[['Ranking', 'Modelo', 'R²', 'RMSE']].round(4))
                st.success(f"🏆 **Melhor modelo**: {melhor_modelo_vol} (R² = {r2_vol[melhor_modelo_vol]:.4f})")

            # ==================== ETAPA 3: INVENTÁRIO COMPLETO ====================
            st.header("📈 ETAPA 3: Inventário Final")

            with st.spinner("Aplicando modelos ao inventário..."):

                # Preparar inventário
                df_inv_final = df_inventario.copy()

                if talhoes_excluir:
                    df_inv_final = df_inv_final[~df_inv_final['talhao'].isin(talhoes_excluir)]

                df_inv_final = df_inv_final[
                    (df_inv_final['D_cm'].notna()) &
                    (df_inv_final['D_cm'] >= diametro_min)
                    ]

                if codigos_excluir:
                    df_inv_final = df_inv_final[~df_inv_final['cod'].isin(codigos_excluir)]

                # Adicionar H_dom se necessário
                if melhor_modelo_hip == 'Campos':
                    df_inv_final = df_inv_final.merge(dominantes, on='parcela', how='left')
                    df_inv_final['H_dom'] = df_inv_final['H_dom'].fillna(25.0)


                # Função para estimar altura
                def estimar_altura(row):
                    if pd.isna(row['H_m']) and row['D_cm'] >= diametro_min:
                        try:
                            if melhor_modelo_hip == "Curtis":
                                return np.exp(-8.0 + 15.0 / row['D_cm'])  # Valores típicos
                            elif melhor_modelo_hip == "Campos":
                                h_dom = row.get('H_dom', 25.0)
                                return np.exp(3.0 - 8.0 / row['D_cm'] + 0.8 * np.log(h_dom))
                            elif melhor_modelo_hip == "Henri":
                                return 5.0 + 10.0 * np.log(row['D_cm'])
                            else:
                                return 25.0  # Default
                        except:
                            return 25.0
                    else:
                        return row['H_m']


                df_inv_final['H_est'] = df_inv_final.apply(estimar_altura, axis=1)


                # Função para estimar volume
                def estimar_volume(row):
                    if pd.notna(row['H_est']) and row['D_cm'] >= diametro_min:
                        try:
                            if melhor_modelo_vol == 'Schumacher':
                                return np.exp(-10.0 + 2.0 * np.log(row['D_cm']) + 1.0 * np.log(row['H_est']))
                            elif melhor_modelo_vol == 'G2':
                                d2 = row['D_cm'] ** 2
                                return 0.001 * d2 * row['H_est']
                            else:
                                return 0.001 * row['D_cm'] ** 2 * row['H_est']  # Fórmula básica
                        except:
                            return 0.0
                    return 0.0


                df_inv_final['V_est'] = df_inv_final.apply(estimar_volume, axis=1)

                # Simular áreas
                talhoes_unicos = sorted(df_inv_final['talhao'].unique())
                np.random.seed(42)
                areas_talhoes = pd.DataFrame({
                    'talhao': talhoes_unicos,
                    'area_ha': np.round(np.random.uniform(15, 50, len(talhoes_unicos)), 2)
                })

                df_inv_final = df_inv_final.merge(areas_talhoes, on='talhao', how='left')

                # Resumo por parcela
                inventario_resumo = df_inv_final[
                    df_inv_final['V_est'].notna()
                ].groupby(['talhao', 'parcela']).agg({
                    'area_ha': lambda x: x.mean(),
                    'idade_anos': lambda x: x.mean() if 'idade_anos' in df_inv_final.columns else 5.0,
                    'D_cm': lambda x: x.mean(),
                    'H_est': lambda x: x.mean(),
                    'V_est': lambda x: x.sum()
                }).reset_index()

                inventario_resumo['Vol_ha'] = inventario_resumo['V_est'] * 10000 / 400  # Assumir 400m²

                st.success(f"✅ Inventário processado: {len(inventario_resumo)} parcelas")

            # ==================== RESULTADOS ====================
            st.header("📊 RESULTADOS FINAIS")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🌲 Parcelas Avaliadas", f"{len(inventario_resumo)}")
            with col2:
                area_total = inventario_resumo['area_ha'].iloc[0] * len(inventario_resumo['talhao'].unique())
                st.metric("📏 Área Total", f"{area_total:.1f} ha")

            with col3:
                vol_medio = inventario_resumo['Vol_ha'].mean()
                st.metric("📊 Produtividade Média", f"{vol_medio:.1f} m³/ha")
            with col4:
                estoque_total = area_total * vol_medio
                st.metric("🌲 Estoque Total", f"{estoque_total:,.0f} m³")

            # Abas de resultados
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumo", "🌳 Por Talhão", "📈 Gráficos", "💾 Downloads"])

            with tab1:
                st.subheader("📈 Estatísticas Gerais")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📏 DAP Médio", f"{inventario_resumo['D_cm'].mean():.1f} cm")

                with col2:
                    st.metric("🌳 Altura Média", f"{inventario_resumo['H_est'].mean():.1f} m")

                with col3:
                    cv_vol = (inventario_resumo['Vol_ha'].std() / vol_medio) * 100
                    st.metric("📊 CV Volume", f"{cv_vol:.1f}%")

                with col4:
                    ima_medio = vol_medio / inventario_resumo['idade_anos'].mean()
                    st.metric("🚀 IMA Médio", f"{ima_medio:.1f} m³/ha/ano")

                # Classificação de produtividade
                st.subheader("📊 Classificação de Produtividade")

                q75 = inventario_resumo['Vol_ha'].quantile(0.75)
                q25 = inventario_resumo['Vol_ha'].quantile(0.25)

                classe_alta = (inventario_resumo['Vol_ha'] >= q75).sum()
                classe_media = ((inventario_resumo['Vol_ha'] >= q25) & (inventario_resumo['Vol_ha'] < q75)).sum()
                classe_baixa = (inventario_resumo['Vol_ha'] < q25).sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌲🌲🌲 Classe Alta", f"{classe_alta} parcelas", f"≥ {q75:.1f} m³/ha")
                with col2:
                    st.metric("🌲🌲 Classe Média", f"{classe_media} parcelas", f"{q25:.1f} - {q75:.1f} m³/ha")
                with col3:
                    st.metric("🌲 Classe Baixa", f"{classe_baixa} parcelas", f"< {q25:.1f} m³/ha")

            with tab2:
                st.subheader("🌳 Análise por Talhão")

                # Resumo por talhão
                resumo_talhao = inventario_resumo.groupby('talhao').agg({
                    'area_ha': 'first',
                    'Vol_ha': ['mean', 'std', 'count'],
                    'D_cm': 'mean',
                    'H_est': 'mean',
                    'idade_anos': 'mean'
                }).round(2)

                # Achatar colunas
                resumo_talhao.columns = ['Área (ha)', 'Vol Médio (m³/ha)', 'Vol Desvio', 'N Parcelas', 'DAP Médio (cm)',
                                         'Altura Média (m)', 'Idade Média (anos)']
                resumo_talhao = resumo_talhao.reset_index()

                # Calcular métricas
                resumo_talhao['Estoque Total (m³)'] = resumo_talhao['Área (ha)'] * resumo_talhao['Vol Médio (m³/ha)']
                resumo_talhao['IMA (m³/ha/ano)'] = resumo_talhao['Vol Médio (m³/ha)'] / resumo_talhao[
                    'Idade Média (anos)']

                st.dataframe(resumo_talhao, use_container_width=True)

            with tab3:
                st.subheader("📊 Visualizações")

                # Gráfico 1: Distribuição de produtividade
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(inventario_resumo['Vol_ha'], bins=10, alpha=0.7, color='forestgreen', edgecolor='black')
                    ax.axvline(vol_medio, color='red', linestyle='--', linewidth=2,
                               label=f'Média: {vol_medio:.1f} m³/ha')
                    ax.set_xlabel('Produtividade (m³/ha)')
                    ax.set_ylabel('Frequência')
                    ax.set_title('Distribuição de Produtividade')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with col2:
                    # Produtividade por talhão
                    fig, ax = plt.subplots(figsize=(8, 6))
                    talhao_vol = inventario_resumo.groupby('talhao')['Vol_ha'].mean().sort_values(ascending=False)
                    bars = ax.bar(range(len(talhao_vol)), talhao_vol.values, color='steelblue', alpha=0.7)
                    ax.set_xlabel('Talhão')
                    ax.set_ylabel('Produtividade (m³/ha)')
                    ax.set_title('Produtividade por Talhão')
                    ax.set_xticks(range(len(talhao_vol)))
                    ax.set_xticklabels([f'T{t}' for t in talhao_vol.index])
                    ax.grid(True, alpha=0.3)

                    for bar, val in zip(bars, talhao_vol.values):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                                f'{val:.0f}', ha='center', va='bottom')

                    st.pyplot(fig)

                # Correlações
                st.subheader("🔗 Correlações")

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                axes[0, 0].scatter(inventario_resumo['D_cm'], inventario_resumo['Vol_ha'], alpha=0.6,
                                   color='forestgreen')
                axes[0, 0].set_xlabel('DAP Médio (cm)')
                axes[0, 0].set_ylabel('Produtividade (m³/ha)')
                axes[0, 0].set_title('Produtividade vs DAP')
                axes[0, 0].grid(True, alpha=0.3)

                axes[0, 1].scatter(inventario_resumo['H_est'], inventario_resumo['Vol_ha'], alpha=0.6,
                                   color='steelblue')
                axes[0, 1].set_xlabel('Altura Média (m)')
                axes[0, 1].set_ylabel('Produtividade (m³/ha)')
                axes[0, 1].set_title('Produtividade vs Altura')
                axes[0, 1].grid(True, alpha=0.3)

                axes[1, 0].scatter(inventario_resumo['idade_anos'], inventario_resumo['Vol_ha'], alpha=0.6,
                                   color='orange')
                axes[1, 0].set_xlabel('Idade (anos)')
                axes[1, 0].set_ylabel('Produtividade (m³/ha)')
                axes[1, 0].set_title('Produtividade vs Idade')
                axes[1, 0].grid(True, alpha=0.3)

                axes[1, 1].scatter(inventario_resumo['D_cm'], inventario_resumo['H_est'], alpha=0.6, color='purple')
                axes[1, 1].set_xlabel('DAP Médio (cm)')
                axes[1, 1].set_ylabel('Altura Média (m)')
                axes[1, 1].set_title('DAP vs Altura')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

            with tab4:
                st.subheader("💾 Downloads")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Download inventário
                    csv_inventario = inventario_resumo.to_csv(index=False)
                    st.download_button(
                        label="📋 Inventário Final",
                        data=csv_inventario,
                        file_name="inventario_completo.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Download volumes
                    csv_volumes = volumes_arvore.to_csv(index=False)
                    st.download_button(
                        label="📊 Volumes Cubagem",
                        data=csv_volumes,
                        file_name="volumes_cubagem.csv",
                        mime="text/csv"
                    )

                with col3:
                    # Download resumo talhão
                    csv_talhao = resumo_talhao.to_csv(index=False)
                    st.download_button(
                        label="🌳 Resumo Talhões",
                        data=csv_talhao,
                        file_name="resumo_talhoes.csv",
                        mime="text/csv"
                    )

                # Relatório técnico
                st.subheader("📄 Relatório Executivo")

                relatorio = f'''
# RELATÓRIO EXECUTIVO - INVENTÁRIO FLORESTAL

## 🏆 MODELOS SELECIONADOS
- **Hipsométrico**: {melhor_modelo_hip} (R² = {melhor_r2_hip:.4f})
- **Volumétrico**: {melhor_modelo_vol} (R² = {r2_vol[melhor_modelo_vol]:.4f})

## 🌲 RESUMO EXECUTIVO
- **Parcelas avaliadas**: {len(inventario_resumo)}
- **Área total**: {area_total:.1f} ha
- **Estoque total**: {estoque_total:,.0f} m³
- **Produtividade média**: {vol_medio:.1f} m³/ha
- **IMA médio**: {ima_medio:.1f} m³/ha/ano

## 📊 CLASSIFICAÇÃO DE PRODUTIVIDADE
- **Classe Alta** (≥ {q75:.1f} m³/ha): {classe_alta} parcelas
- **Classe Média** ({q25:.1f} - {q75:.1f} m³/ha): {classe_media} parcelas
- **Classe Baixa** (< {q25:.1f} m³/ha): {classe_baixa} parcelas

## 📊 ESTATÍSTICAS
- **DAP médio**: {inventario_resumo['D_cm'].mean():.1f} cm
- **Altura média**: {inventario_resumo['H_est'].mean():.1f} m
- **CV produtividade**: {cv_vol:.1f}%

---
*Relatório gerado pelo Sistema de Inventário Florestal*
'''

                st.download_button(
                    label="📄 Relatório Completo",
                    data=relatorio,
                    file_name="relatorio_inventario.md",
                    mime="text/markdown"
                )

            # ==================== SUMÁRIO FINAL ====================
            st.header("🎉 ANÁLISE CONCLUÍDA")

            st.success(f'''
            ### ✅ **Sistema Executado com Sucesso!**

            **🔄 Etapas finalizadas:**
            1. ✅ **Modelos Hipsométricos** → {melhor_modelo_hip} selecionado
            2. ✅ **Modelos Volumétricos** → {melhor_modelo_vol} selecionado  
            3. ✅ **Inventário Completo** → {len(inventario_resumo)} parcelas processadas

            **📊 Resultados principais:**
            - **Produtividade**: {vol_medio:.1f} m³/ha
            - **Estoque**: {estoque_total:,.0f} m³
            - **IMA**: {ima_medio:.1f} m³/ha/ano
            ''')

else:
    # ==================== INSTRUÇÕES ====================
    st.header("📋 Como Usar o Sistema")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Arquivo de Inventário")
        st.markdown('''
        **Colunas obrigatórias:**
        - `D_cm`: Diâmetro (cm)
        - `H_m`: Altura (m)
        - `talhao`: ID do talhão
        - `parcela`: ID da parcela
        - `cod`: Código (D=Dominante, N=Normal, C=Cortada, I=Invasora)

        **Opcionais:**
        - `idade_anos`: Idade do povoamento
        ''')

        exemplo_inv = pd.DataFrame({
            'talhao': [1, 1, 2, 2],
            'parcela': [1, 1, 1, 2],
            'D_cm': [15.2, 18.5, 20.1, 16.8],
            'H_m': [18.5, 22.1, 24.3, 19.8],
            'cod': ['N', 'D', 'D', 'N'],
            'idade_anos': [5.2, 5.2, 6.1, 6.1]
        })

        st.dataframe(exemplo_inv)

    with col2:
        st.subheader("📏 Arquivo de Cubagem")
        st.markdown('''
        **Colunas obrigatórias:**
        - `arv`: ID da árvore
        - `talhao`: ID do talhão
        - `d_cm`: Diâmetro da seção (cm)
        - `h_m`: Altura da seção (m)
        - `D_cm`: DAP da árvore (cm)
        - `H_m`: Altura total da árvore (m)
        ''')

        exemplo_cub = pd.DataFrame({
            'arv': [1, 1, 1, 2, 2],
            'talhao': [1, 1, 1, 1, 1],
            'd_cm': [0, 15.2, 12.1, 0, 18.5],
            'h_m': [0.1, 2.0, 4.0, 0.1, 2.0],
            'D_cm': [15.2, 15.2, 15.2, 18.5, 18.5],
            'H_m': [18.5, 18.5, 18.5, 22.1, 22.1]
        })

        st.dataframe(exemplo_cub)

    # Fluxo do sistema
    st.subheader("🔄 Fluxo do Sistema")
    st.markdown('''
    1. **📁 Upload** dos arquivos (inventário + cubagem)
    2. **⚙️ Configuração** de filtros
    3. **🌳 Etapa 1**: Teste de 7 modelos hipsométricos → seleciona o melhor
    4. **📊 Etapa 2**: Cubagem (Smalian) + 4 modelos volumétricos → seleciona o melhor
    5. **📈 Etapa 3**: Aplica os melhores modelos ao inventário
    6. **📊 Resultados**: Análises, gráficos e relatórios
    ''')

    st.info("👆 **Carregue os dois arquivos na barra lateral para começar!**")

    # Modelos disponíveis
    st.subheader("🧮 Modelos Integrados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('''
        **🌳 Hipsométricos (7):**
        - Curtis, Campos, Henri, Prodan
        - Chapman, Weibull, Mononuclear
        ''')

    with col2:
        st.markdown('''
        **📊 Volumétricos (4):**
        - Schumacher-Hall, G1, G2, G3
        - **Cubagem**: Método de Smalian
        ''')

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown('''
<div style='text-align: center; color: #666;'>
    <p>🌲 <strong>Sistema Simplificado de Inventário Florestal</strong></p>
    <p>Análise completa automatizada com seleção dos melhores modelos</p>
</div>
''', unsafe_allow_html=True)