# 🌲 Sistema Completo Integrado de Inventário Florestal

Aplicação web avançada para análise completa de inventários florestais, integrando modelos hipsométricos, volumétricos e estimativas de produção florestal em um sistema unificado.

## 📋 Descrição

Este sistema revoluciona a análise de inventários florestais ao integrar **três etapas sequenciais** em uma única aplicação:

1. **🌳 Análise Hipsométrica** - Testa 7 modelos e seleciona automaticamente o melhor
2. **📊 Análise Volumétrica** - Processa cubagem e compara 4 modelos de volume
3. **📈 Inventário Integrado** - Aplica os melhores modelos e gera análises completas

O diferencial é a **seleção automática** dos melhores modelos e **integração fluida** entre as etapas, eliminando a necessidade de múltiplas ferramentas.

## ✨ Funcionalidades Principais

### 📁 **Upload e Processamento Inteligente**
- Suporte **robusto** para arquivos **CSV**, **Excel** (.xlsx, .xls, .xlsb)
- **Detecção automática** de separadores e engines Excel
- **Validação completa** das colunas necessárias
- **Filtros configuráveis** para talhões, diâmetro mínimo e códigos de árvores
- **Tratamento inteligente** de dados faltantes e outliers

### 📊 **ETAPA 1: Modelos Hipsométricos (7 Modelos)**

#### Modelos Lineares
- **Curtis**: `ln(H) = β₀ + β₁ × (1/D)`
- **Campos**: `ln(H) = β₀ + β₁ × (1/D) + β₂ × ln(H_dom)`
- **Henri**: `H = β₀ + β₁ × ln(D)`
- **Prodan**: `D²/(H-1.3) = β₀ + β₁ × D + β₂ × D² + β₃ × D × Idade`

#### Modelos Não-Lineares
- **Chapman-Richards**: `H = b₀ × (1 - exp(-b₁ × D))^b₂`
- **Weibull**: `H = a × (1 - exp(-b × D^c))`
- **Mononuclear**: `H = a × (1 - b × exp(-c × D))`

### 📊 **ETAPA 2: Modelos Volumétricos (4 Modelos)**
- **Schumacher-Hall**: `ln(V) = β₀ + β₁×ln(D) + β₂×ln(H)`
- **G1**: `ln(V) = β₀ + β₁×ln(D) + β₂×(1/D)`
- **G2**: `V = β₀ + β₁×D² + β₂×D²H + β₃×H`
- **G3**: `ln(V) = β₀ + β₁×ln(D²H)`

**Método de Cubagem**: Fórmula de Smalian com exclusão automática do toco

### 🔧 **Análise Detalhada por Modelo**
- **Abas individuais** para cada modelo com:
  - **Equações LaTeX** profissionais
  - **Coeficientes detalhados** com significância estatística
  - **Gráficos específicos** (observado vs predito)
  - **Análise de resíduos** (scatter + histograma)
  - **Classificação qualitativa** (Excelente, Muito Bom, Bom, Regular, Fraco)
  - **Ranking automático** por performance

### 📈 **ETAPA 3: Inventário Integrado**
- **Aplicação automática** dos melhores modelos selecionados
- **Estimativas por parcela** com volumes e produtividade
- **Análise por talhão** com estatísticas consolidadas
- **Classificação de produtividade** automática (Alta, Média, Baixa)
- **Cálculo de IMA** (Incremento Médio Anual - m³/ha/ano)

### 📊 **Métricas de Avaliação Avançadas**
- **R² Generalizado** para modelos hipsométricos
- **R² Padrão** para modelos volumétricos
- **RMSE** (Raiz do Erro Quadrático Médio)
- **MAE** (Erro Absoluto Médio)
- **Estatística F** e **significância** para modelos lineares
- **Rankings automáticos** com classificação qualitativa

### 📊 **Visualizações Interativas**
- **Gráficos comparativos** de todos os modelos
- **Análise de resíduos** detalhada por modelo
- **Distribuição de produtividade** por talhão
- **Correlações múltiplas** (DAP vs Altura vs Volume vs Idade)
- **Classificação visual** de sítios

### 💾 **Exportação Completa de Resultados**
- **Inventário final** com estimativas (CSV)
- **Dados de cubagem** processados (CSV)
- **Resumo por talhão** com métricas (CSV)
- **Coeficientes dos modelos** com significância (CSV)
- **Relatório técnico executivo** completo (Markdown)

## 🚀 Instalação e Uso

### Pré-requisitos
- **Python 3.12.3** (recomendado) ou superior
- Bibliotecas especificadas no `requirements.txt`

### Instalação Rápida

1. **Clone o repositório:**
```bash
git clone <url-do-repositorio>
cd sistema-inventario-florestal
```

2. **Crie um ambiente virtual:**
```bash
python -m venv inventario_env
source inventario_env/bin/activate  # Linux/Mac
# ou
inventario_env\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

### Executando o Sistema

```bash
streamlit run app.py
```

O sistema abrirá automaticamente no navegador em `http://localhost:8501`

## 📊 Formato dos Dados

### 📋 **Arquivo de Inventário (Obrigatório)**

**Colunas obrigatórias:**
- `D_cm`: Diâmetro à altura do peito (cm)
- `H_m`: Altura total (m)
- `talhao`: Identificação do talhão
- `parcela`: Identificação da parcela
- `cod`: Código da árvore
  - `D` = Dominante
  - `N` = Normal
  - `C` = Cortada
  - `I` = Invasora

**Colunas opcionais:**
- `idade_anos`: Idade do povoamento (necessária para modelo Prodan completo)

### 📏 **Arquivo de Cubagem (Obrigatório)**

**Colunas obrigatórias:**
- `arv`: Identificação única da árvore
- `talhao`: Identificação do talhão
- `d_cm`: Diâmetro da seção (cm)
- `h_m`: Altura da seção (m)
- `D_cm`: DAP da árvore (cm)
- `H_m`: Altura total da árvore (m)

### Exemplo de Arquivo de Inventário (CSV)
```csv
talhao,parcela,D_cm,H_m,cod,idade_anos
1,1,15.2,18.5,N,5.2
1,1,18.5,22.1,D,5.2
1,2,12.3,16.2,N,5.2
2,1,20.1,24.3,D,6.1
2,1,16.8,19.8,N,6.1
```

### Exemplo de Arquivo de Cubagem (CSV)
```csv
arv,talhao,d_cm,h_m,D_cm,H_m
1,1,0,0.1,15.2,18.5
1,1,15.2,2.0,15.2,18.5
1,1,12.1,4.0,15.2,18.5
2,1,0,0.1,18.5,22.1
2,1,18.5,2.0,18.5,22.1
```

## 🔍 Como Usar o Sistema

### **Passo 1: Upload dos Dados**
1. Acesse a barra lateral esquerda
2. Faça upload do **Arquivo de Inventário**
3. Faça upload do **Arquivo de Cubagem**
4. O sistema validará automaticamente as colunas

### **Passo 2: Configurações**
1. **Talhões a excluir**: Selecione áreas experimentais ou Pinus
2. **Diâmetro mínimo**: Defina critério mínimo (padrão: 4.0 cm)
3. **Códigos a excluir**: Remova árvores cortadas (C) ou invasoras (I)

### **Passo 3: Execução**
1. Clique em **"🚀 Executar Análise Completa"**
2. O sistema processará automaticamente:
   - **ETAPA 1**: Teste dos 7 modelos hipsométricos
   - **ETAPA 2**: Cubagem + 4 modelos volumétricos
   - **ETAPA 3**: Inventário final integrado

### **Passo 4: Análise dos Resultados**
1. **Explore as abas** de cada modelo individual
2. **Analise gráficos** e coeficientes detalhados
3. **Compare rankings** automáticos
4. **Revise inventário final** com classificação de produtividade

### **Passo 5: Download**
1. **Baixe relatórios** em múltiplos formatos
2. **Exporte dados** processados
3. **Salve coeficientes** para uso futuro

## 📚 Interpretação dos Resultados

### **Classificação de Qualidade dos Modelos**
- **🟢 Excelente**: R² ≥ 0.90
- **🔵 Muito Bom**: 0.80 ≤ R² < 0.90
- **🟡 Bom**: 0.70 ≤ R² < 0.80
- **🟠 Regular**: 0.60 ≤ R² < 0.70
- **🔴 Fraco**: R² < 0.60

### **Classificação de Produtividade**
- **🟢 Classe Alta**: Produtividade ≥ Q75 (quartil superior)
- **🟡 Classe Média**: Q25 ≤ Produtividade < Q75 (quartis intermediários)
- **🔴 Classe Baixa**: Produtividade < Q25 (quartil inferior)

### **Significância Estatística (Modelos Lineares)**
- ***** p < 0.001: Altamente significativo
- **** p < 0.01: Muito significativo
- *** p < 0.05: Significativo
- **. p < 0.1**: Marginalmente significativo
- **(espaço)**: Não significativo

## 🛠️ Dependências Principais

```txt
streamlit>=1.35.0          # Interface web moderna
pandas>=2.2.0              # Manipulação de dados
numpy>=1.26.0              # Operações numéricas
matplotlib>=3.8.0          # Visualizações básicas
seaborn>=0.13.0            # Gráficos estatísticos
scikit-learn>=1.4.0        # Modelos de regressão
scipy>=1.12.0              # Otimização e estatística
openpyxl>=3.1.2            # Arquivos Excel (.xlsx)
xlrd>=2.0.1                # Arquivos Excel (.xls)
pyxlsb>=1.0.10             # Arquivos Excel (.xlsb)
```

### **Instalação de Dependências Excel**
Se houver problemas com arquivos Excel:
```bash
pip install openpyxl xlrd pyxlsb
```

**Alternativa**: Converta arquivos Excel para CSV:
- Excel → Arquivo → Salvar Como → CSV UTF-8

## 📖 Conceitos Importantes

### 🏞️ **Diferença: Sítio vs Classificação de Produtividade**

**SÍTIO FLORESTAL (Índice de Local)**
- **Conceito**: Capacidade produtiva **inerente** do local
- **Baseado em**: Características edafoclimáticas (solo, clima, topografia)
- **Método**: Altura dominante × idade (curvas de índice de sítio)
- **Característica**: Propriedade **permanente** do local
- **Aplicação**: Planejamento de longo prazo e seleção de espécies

**CLASSIFICAÇÃO DE PRODUTIVIDADE (Este Sistema)**
- **Conceito**: Performance **atual** observada no inventário
- **Baseado em**: Volume/hectare medido nas parcelas
- **Método**: Estratificação por quartis (Q25, Q75)
- **Característica**: Pode **variar** com manejo e idade
- **Aplicação**: Análise de desempenho atual e estratificação operacional

### 🌳 **Métricas Florestais Importantes**

**IMA (Incremento Médio Anual)**
- **Fórmula**: `IMA = Volume atual (m³/ha) ÷ Idade (anos)`
- **Significado**: Produtividade média anual do povoamento
- **Unidade**: m³/ha/ano
- **Aplicação**: Comparação de produtividade entre talhões e idades
- **Exemplo**: Se um talhão de 6 anos tem 180 m³/ha → IMA = 30 m³/ha/ano

**ICA (Incremento Corrente Anual)**
- **Conceito**: Crescimento no último ano
- **Relação**: Quando ICA = IMA → idade de rotação ótima
- **Aplicação**: Determinação do momento ideal de corte

### 🌳 **Talhão vs Parcela**
- **Talhão**: Unidade de manejo florestal (homogênea em idade/espécie)
- **Parcela**: Unidade de amostragem do inventário (400m² típico)

*O sistema classifica **parcelas por produtividade**, não determina índice de sítio florestal.*

## 📖 Fundamentos Científicos
- **Campos, J.C.C. & Leite, H.G.** - Mensuração Florestal: Perguntas e Respostas
- **Scolforo, J.R.S.** - Biometria Florestal
- **Burkhart, H.E. & Tomé, M.** - Modeling Forest Trees and Stands

### **Modelos Volumétricos**
- **Husch, B., Beers, T.W. & Kershaw Jr., J.A.** - Forest Mensuration
- **West, P.W.** - Tree and Forest Measurement

### **Metodologia Estatística**
- **Ratkowsky, D.A.** - Handbook of Nonlinear Regression Models
- **Draper, N.R. & Smith, H.** - Applied Regression Analysis

## 🌟 Características Avançadas

### **🔄 Fluxo Integrado Único**
- **Seleção automática** dos melhores modelos
- **Integração fluida** entre etapas
- **Validação cruzada** automática
- **Diagnóstico de qualidade** integrado

### **📊 Interface Profissional**
- **Abas organizadas** por modelo
- **Equações LaTeX** matematicamente corretas
- **Gráficos de alta qualidade** para publicação
- **Relatórios executivos** prontos para gestão

### **🔍 Análise Estatística Robusta**
- **Múltiplas métricas** de avaliação
- **Análise de resíduos** detalhada
- **Teste de significância** para modelos lineares
- **Rankings automáticos** por performance

### **💾 Exportação Versátil**
- **Múltiplos formatos** (CSV, Markdown)
- **Relatórios técnicos** completos
- **Dados processados** para uso posterior
- **Coeficientes** para implementação operacional

### **📊 Classificação Inteligente**
- **Estratificação por produtividade** baseada em quartis
- **Classes Alta/Média/Baixa** automaticamente definidas
- **Análise por talhão** com métricas consolidadas
- **IMA e produtividade** calculados automaticamente

## 🤝 Contribuições

Contribuições são muito bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

### **Áreas para Contribuição**
- 🌱 Novos modelos hipsométricos/volumétricos
- 📊 Melhorias nas visualizações
- 🔧 Otimização de performance
- 📝 Documentação e exemplos
- 🧪 Testes automatizados
- 🌍 Internacionalização

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📧 Suporte e Contato

Para dúvidas, sugestões ou problemas:
- 📧 **Email**: [seu-email@exemplo.com]
- 🐛 **Issues**: Abra uma issue no GitHub
- 💬 **Discussões**: Use as Discussions do GitHub
- 📖 **Wiki**: Consulte a documentação completa

## 🔄 Changelog

### v2.0.0 - Sistema Integrado (Atual)
- ✅ **Integração completa** das 3 etapas
- ✅ **7 modelos hipsométricos** + 4 volumétricos
- ✅ **Análise detalhada** por modelo individual
- ✅ **Seleção automática** dos melhores modelos
- ✅ **Interface moderna** com abas organizadas
- ✅ **Relatórios executivos** completos
- ✅ **Múltiplas engines Excel** para máxima compatibilidade

### v1.0.0 - Modelos Hipsométricos
- ✅ Sistema básico de modelos hipsométricos
- ✅ Interface Streamlit inicial
- ✅ Exportação de resultados

---

**🌲 Desenvolvido para a comunidade florestal brasileira com foco em excelência técnica e facilidade de uso** 🇧🇷

*Sistema que une rigor científico, interface moderna e praticidade operacional para revolucionar a análise de inventários florestais.*
