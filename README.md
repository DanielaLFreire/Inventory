# 🌲 Sistema Integrado de Inventário Florestal

Sistema modular para análise completa de inventário florestal com modelos hipsométricos e volumétricos.

## 🚀 Funcionalidades
📁 Upload de Dados
    ↓
⚙️ Etapa 0: Configurações (NOVA)
    ├── 🔍 Filtros de Dados
    ├── 📏 Áreas dos Talhões
    ├── 🌱 Parâmetros Florestais
    └── 🧮 Configurações de Modelos
    ↓
🌳 Etapa 1: Hipsométricos (USA CONFIG GLOBAL)
   **🌳 Modelos Hipsométricos** (7 modelos)
   - Curtis, Campos, Henri, Prodan
   - Chapman, Weibull, Mononuclear
     - Seleção automática do melhor model
      ↓
📊 Etapa 2: Volumétricos (USA CONFIG GLOBAL)
   **📏 Modelos Volumétricos** (4 modelos + Cubagem)
   - Schumacher-Hall, G1, G2, G3
   - Cubagem pelo método de Smalian
     - Seleção automática do melhor modelo
      ↓
📈 Etapa 3: Inventário (USA CONFIG GLOBAL)
   **📈 Inventário Final**
   - Aplicação dos melhores modelos
   - Cálculos de produtividade
   - Relatórios e visualizações


## 📁 Estrutura do Projeto

```
inventario_florestal/
├── app.py                    # Aplicação principal
├── requirements.txt          # Dependências
├── README.md                # Documentação
│
├── config/
│   └── config.py            # Configurações globais
│
├── utils/
│   ├── formatacao.py        # Formatação brasileira
│   ├── validacao.py         # Validação de dados
│   └── arquivo_handler.py   # Manipulação de arquivos
│
├── models/
│   ├── base.py              # Classes base
│   ├── hipsometrico.py      # Modelos hipsométricos
│   └── volumetrico.py       # Modelos volumétricos
│
├── processors/
│   ├── cubagem.py           # Processamento de cubagem
│   ├── inventario.py        # Processamento do inventário
│   └── areas.py             # Processamento de áreas
│
└── ui/
    ├── sidebar.py           # Interface lateral
    ├── configuracoes.py     # Configurações
    ├── resultados.py        # Resultados
    └── graficos.py          # Visualizações
```

## 🛠️ Instalação

### 1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/inventario-florestal.git
cd inventario-florestal
```

### 2. **Crie ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. **Instale dependências**
```bash
pip install -r requirements.txt
```

### 4. **Execute a aplicação**
```bash
streamlit run app.py
```

## 📋 Formato dos Dados

### 📊 **Arquivo de Inventário**
Colunas obrigatórias:
- `D_cm`: Diâmetro (cm)
- `H_m`: Altura (m) 
- `talhao`: ID do talhão
- `parcela`: ID da parcela
- `cod`: Código (D=Dominante, N=Normal, C=Cortada, I=Invasora)

Opcionais:
- `idade_anos`: Idade do povoamento

### 📏 **Arquivo de Cubagem**
Colunas obrigatórias:
- `arv`: ID da árvore
- `talhao`: ID do talhão
- `d_cm`: Diâmetro da seção (cm)
- `h_m`: Altura da seção (m)
- `D_cm`: DAP da árvore (cm)
- `H_m`: Altura total da árvore (m)

## 🔧 Configurações

### **Filtros Disponíveis**
- Talhões a excluir
- Diâmetro mínimo (cm)
- Códigos a excluir
- Área da parcela (m²)

### **Métodos de Área**
- Simulação automática
- Valores manuais
- Upload de shapefile
- Coordenadas das parcelas

## 📊 Modelos Implementados

### 🌳 **Hipsométricos**

| Modelo | Equação | Tipo |
|--------|---------|------|
| Curtis | ln(H) = β₀ + β₁ × (1/D) | Linear |
| Campos | ln(H) = β₀ + β₁ × (1/D) + β₂ × ln(H_dom) | Linear |
| Henri | H = β₀ + β₁ × ln(D) | Linear |
| Prodan | D²/(H-1.3) = β₀ + β₁×D + β₂×D² | Linear |
| Chapman | H = b₀ × (1 - e^(-b₁×D))^b₂ | Não-linear |
| Weibull | H = a × (1 - e^(-b×D^c)) | Não-linear |
| Mononuclear | H = a × (1 - b×e^(-c×D)) | Não-linear |

### 📏 **Volumétricos**

| Modelo | Equação | Tipo |
|--------|---------|------|
| Schumacher | ln(V) = β₀ + β₁×ln(D) + β₂×ln(H) | Linear |
| G1 | ln(V) = β₀ + β₁×ln(D) + β₂×(1/D) | Linear |
| G2 | V = β₀ + β₁×D² + β₂×D²H + β₃×H | Linear |
| G3 | ln(V) = β₀ + β₁×ln(D²H) | Linear |

## 📈 Resultados

### **Métricas de Avaliação**
- **R² Generalizado** (modelos hipsométricos)
- **R² tradicional** (modelos volumétricos)
- **RMSE** (Root Mean Square Error)
- **Ranking automático** dos modelos

### **Outputs Gerados**
- Inventário completo processado
- Resumo por talhão e parcela
- Gráficos de ajuste e resíduos
- Relatório executivo em Markdown
- Arquivos CSV para download

## 🎯 Vantagens da Versão Modular

✅ **Modularidade**: Cada funcionalidade em módulo específico  
✅ **Manutenibilidade**: Fácil localização e correção  
✅ **Escalabilidade**: Simples adicionar novos modelos  
✅ **Testabilidade**: Módulos independentes  
✅ **Reutilização**: Código reutilizável  
✅ **Clareza**: Organização clara e lógica  

## 🔧 Personalização

### **Adicionar Novo Modelo Hipsométrico**
1. Crie classe herdando de `ModeloLinear` ou `ModeloNaoLinear`
2. Implemente métodos `preparar_dados()` e `predizer_altura()`
3. Adicione à lista em `ajustar_todos_modelos_hipsometricos()`

### **Adicionar Novo Modelo Volumétrico**
1. Crie classe herdando de `ModeloLinear`
2. Implemente métodos `preparar_dados()` e `predizer_volume()`
3. Adicione à lista em `ajustar_todos_modelos_volumetricos()`

## 🐛 Resolução de Problemas

### **Erro de Engine Excel**
```bash
pip install openpyxl xlrd pyxlsb
```

### **Erro GeoPandas (Shapefile)**
```bash
pip install geopandas shapely
```

### **Problemas de Encoding**
- Salve CSV em UTF-8
- Use separador ponto-e-vírgula (;)

## 📞 Suporte

- **Issues**: Abra issue no GitHub
- **Documentação**: Consulte este README
- **Exemplos**: Veja pasta `data/exemplos/`

## 📄 Licença

Este projeto está sob licença MIT. Veja arquivo `LICENSE` para detalhes.

---

**🌲 Sistema Modular de Inventário Florestal**  
*Análise completa automatizada com seleção dos melhores modelos*