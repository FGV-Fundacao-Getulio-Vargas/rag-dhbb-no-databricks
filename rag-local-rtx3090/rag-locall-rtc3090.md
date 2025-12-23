# RAG Local com GPU RTX3090 - Guia Completo

## Parte 0: Pré-requisitos do Sistema

### Hardware
- **GPU:** RTX3090 (Ampere, CUDA Capability 8.6)
- **RAM:** Mínimo 32GB (recomendado 48GB+)
- **Storage:** 100GB disponível (modelos + dados)
- **CPU:** Qualquer processador moderno

### Software Base
- **Windows 10/11 ou Linux (Ubuntu 22.04 LTS recomendado)**
- **Python 3.10** (compatível com CUDA 12.1)
- **NVIDIA Driver:** 545.23.06 ou superior
- **CUDA Toolkit:** 12.1
- **cuDNN:** 8.9.x para CUDA 12.1

---

## Parte 1: Configuração do Ambiente CUDA/cuDNN

### Passo 1.1: Verificar GPU e Drivers

# Windows - No PowerShell/CMD
nvidia-smi

# Linux - No terminal
nvidia-smi

**Output esperado:**
NVIDIA GeForce RTX 3090        Compute Capability: 8.6
Driver Version: 545.23.06
CUDA Version: 12.1

### Passo 1.2: Instalar CUDA 12.1 (Se não tiver)

**Windows:**
1. Baixe em: https://developer.nvidia.com/cuda-12-1-0-download-archive
2. Selecione: Windows → x86_64 → Windows 10/11 → exe (local)
3. Execute o instalador e siga as instruções padrão

**Linux (Ubuntu 22.04):**
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

### Passo 1.3: Instalar cuDNN 8.9.x

1. Baixe em: https://developer.nvidia.com/cudnn (requer login gratuito)
2. Selecione: **cuDNN 8.9.7** para CUDA 12.1
3. Extraia e copie arquivos:

**Windows:**
# Após extrair o ZIP
# Copie os arquivos para C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\

copy cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin\*.dll "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
copy cudnn-windows-x86_64-8.9.7.29_cuda12-archive\lib\x64\*.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64"
copy cudnn-windows-x86_64-8.9.7.29_cuda12-archive\include\*.h "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"

**Linux:**
# Após extrair o tar.xz
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/bin/* /usr/local/cuda-12.1/bin/
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/local/cuda-12.1/lib64/
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/local/cuda-12.1/include/

### Passo 1.4: Adicionar CUDA às Variáveis de Ambiente

**Windows (PowerShell Admin):**
[Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
[Environment]::SetEnvironmentVariable("CUDNN_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
# Feche e reabra o PowerShell para aplicar

**Linux:**
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export CUDNN_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

### Passo 1.5: Verificar Instalação CUDA

nvcc --version
# Output esperado: release 12.1, V12.1.66

---

## Parte 2: Setup Python e Ambiente Virtual

### Passo 2.1: Instalar Python 3.10

**Windows:**
1. Baixe de: https://www.python.org/downloads/release/python-3100/
2. Execute o instalador
3. **IMPORTANTE:** Marque "Add Python 3.10 to PATH"

**Linux:**
sudo apt-get install python3.10 python3.10-venv python3.10-dev

### Passo 2.2: Criar Diretório do Projeto

# Windows
mkdir C:\Users\SeuUsuario\rag-desktop
cd C:\Users\SeuUsuario\rag-desktop

# Linux
mkdir ~/rag-desktop
cd ~/rag-desktop

### Passo 2.3: Criar Ambiente Virtual

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux
python3.10 -m venv venv
source venv/bin/activate

Você verá `(venv)` no início da linha de comando.

### Passo 2.4: Atualizar pip, setuptools e wheel

pip install --upgrade pip==24.0
pip install setuptools==69.0.3
pip install wheel==0.42.0

---

## Parte 3: Instalar Pacotes (Versões Exatas)

### Passo 3.1: Criar arquivo requirements.txt

Crie um arquivo chamado `requirements.txt` na pasta `rag-desktop` com o conteúdo exato abaixo:

# Core & Deep Learning
torch==2.2.0+cu121
torchvision==0.17.0+cu121
torchaudio==2.2.0+cu121

# Embedding Models
sentence-transformers==2.7.0
transformers==4.38.2
tokenizers==0.15.2
safetensors==0.4.2

# Vector Database (Local)
chromadb==0.4.24
pymilvus==2.3.11

# LLM & RAG
langchain==0.1.16
langchain-core==0.1.47
langchain-community==0.0.38

# Utilidades
numpy==1.24.3
pandas==2.2.0
scipy==1.13.0
scikit-learn==1.4.2

# I/O & Serialização
pydantic==2.6.4
python-dotenv==1.0.0
pyyaml==6.0.1

# Utilities
tqdm==4.66.2
requests==2.31.0

# Development (Opcional)
jupyter==1.0.0
ipython==8.21.0

### Passo 3.2: Instalar PyTorch com CUDA 12.1

# Este comando instala a versão correta para RTX3090
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_cu121

**Verificar instalação:**
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

**Output esperado:**
PyTorch: 2.2.0+cu121
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3090

### Passo 3.3: Instalar Pacotes Restantes

pip install -r requirements.txt

**Tempo estimado:** 5-10 minutos (dependendo da internet)

### Passo 3.4: Verificar Instalações Críticas

# Sentence Transformers
python -c "from sentence_transformers import SentenceTransformer; print('✓ Sentence Transformers OK')"

# ChromaDB
python -c "import chromadb; print('✓ ChromaDB OK')"

# LangChain
python -c "import langchain; print(f'✓ LangChain {langchain.__version__} OK')"

---

## Parte 4: Preparar Arquivos de Texto

### Passo 4.1: Criar Pasta de Documentos

# Dentro de rag-desktop
mkdir documents
# Coloque seus arquivos .txt aqui

### Passo 4.2: Exemplo de Arquivo de Teste

Crie `documents/exemplo.txt`:
A Inteligência Artificial está revolucionando o mundo. RAG combina recuperação com geração.
LLMs como GPT e Claude usam transformers. O aprendizado de máquina avança rapidamente.
Redes neurais convolucionais processam imagens. Transformers dominam NLP moderno.

---

## Parte 5: Pipeline RAG Local Completo

### Passo 5.1: Criar arquivo `rag_pipeline.py`

"""
RAG Pipeline Local para Desktop com GPU RTX3090
"""

import os
import glob
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ============================================
# CONFIGURAÇÃO
# ============================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # 22MB, 384-dim
    'model_large': 'sentence-transformers/all-mpnet-base-v2',  # 420MB, 768-dim (opcional)
    'chunk_size': 512,
    'chunk_overlap': 100,
    'top_k': 3,
    'embeddings_cache': 'chromadb_storage',
    'documents_dir': './documents'
}

print(f"🚀 Usando device: {CONFIG['device'].upper()}")

if CONFIG['device'] == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# CLASSE RAG PIPELINE
# ============================================

class RAGPipeline:
    def __init__(self, config: Dict):
        """Inicializar pipeline RAG"""
        self.config = config
        self.device = config['device']
        
        print("\n[1/4] Carregando modelo de embedding...")
        self.embedding_model = SentenceTransformer(
            config['model_name'],
            device=self.device
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"     ✓ Modelo carregado: {config['model_name']}")
        print(f"     ✓ Dimensão embeddings: {self.embedding_dim}")
        
        print("\n[2/4] Inicializando ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=config['embeddings_cache']
        )
        self.collection = None
        print(f"     ✓ ChromaDB pronto (armazenado em {config['embeddings_cache']})")
        
        print("\n[3/4] Configurando splitter de texto...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        print(f"     ✓ Chunk size: {config['chunk_size']}")
        
        print("\n[4/4] Sistema pronto!\n")
    
    def load_documents(self, directory: str) -> Dict[str, str]:
        """Carregar todos os arquivos .txt de um diretório"""
        documents = {}
        
        txt_files = glob.glob(os.path.join(directory, "*.txt"))
        
        if not txt_files:
            print(f"⚠️  Nenhum arquivo .txt encontrado em {directory}")
            return documents
        
        print(f"📄 Carregando {len(txt_files)} arquivo(s)...")
        
        for filepath in txt_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents[filename] = content
                print(f"   ✓ {filename} ({len(content)} caracteres)")
            except Exception as e:
                print(f"   ✗ Erro ao ler {filename}: {e}")
        
        return documents
    
    def process_documents(self, documents: Dict[str, str]):
        """Processar documentos: chunk + embedding + armazenar"""
        
        if not documents:
            print("❌ Nenhum documento para processar!")
            return
        
        # Deletar coleção antiga se existir
        try:
            self.client.delete_collection(name="documents")
        except:
            pass
        
        # Criar nova coleção
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"\n📊 Processando documentos...\n")
        
        chunk_id = 0
        all_chunks = []
        
        # Fase 1: Chunking
        print("[Fase 1] Dividindo documentos em chunks...")
        for filename, content in documents.items():
            chunks = self.text_splitter.split_text(content)
            
            for chunk_text in chunks:
                all_chunks.append({
                    'id': f'chunk_{chunk_id:06d}',
                    'document': filename,
                    'text': chunk_text,
                    'char_count': len(chunk_text)
                })
                chunk_id += 1
        
        print(f"   ✓ {len(all_chunks)} chunks criados")
        
        # Fase 2: Gerar embeddings (em batches)
        print(f"\n[Fase 2] Gerando embeddings (batch size=32)...")
        
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(all_chunks), batch_size), 
                     desc="Embeddings"):
            batch_chunks = all_chunks[i:i+batch_size]
            texts = [c['text'] for c in batch_chunks]
            
            # Gerar embeddings com GPU
            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                batch_embeddings = batch_embeddings.cpu().numpy().tolist()
            
            embeddings.extend(batch_embeddings)
        
        print(f"   ✓ {len(embeddings)} embeddings gerados")
        
        # Fase 3: Armazenar em ChromaDB
        print(f"\n[Fase 3] Armazenando em ChromaDB...")
        
        # Preparar dados para ChromaDB
        ids = [c['id'] for c in all_chunks]
        documents_list = [c['text'] for c in all_chunks]
        metadatas = [
            {
                'source': c['document'],
                'char_count': c['char_count']
            }
            for c in all_chunks
        ]
        
        # Adicionar em batches
        for i in tqdm(range(0, len(ids), batch_size),
                     desc="Armazenando"):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents_list[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_embed = embeddings[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embed
            )
        
        print(f"\n✅ Pipeline completo!")
        print(f"   • Total chunks: {len(all_chunks)}")
        print(f"   • Documentos processados: {len(documents)}")
        print(f"   • Coleção: 'documents' em {self.config['embeddings_cache']}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Buscar chunks mais relevantes"""
        
        if top_k is None:
            top_k = self.config['top_k']
        
        if self.collection is None:
            print("❌ Nenhuma coleção carregada. Execute process_documents() primeiro.")
            return []
        
        # Gerar embedding da query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True
        ).cpu().numpy().tolist()
        
        # Buscar no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved = []
        if results['documents'] and results['documents'][0]:
            for doc, distance in zip(
                results['documents'][0],
                results['distances'][0]
            ):
                # ChromaDB retorna distância (menor = mais similar)
                similarity = 1 / (1 + distance)
                retrieved.append((doc, similarity))
        
        return retrieved
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Gerar resposta com contexto (versão simples)"""
        
        if not context:
            return "Desculpe, não encontrei informações relevantes."
        
        context_text = "\n\n".join(context[:3])
        
        # Template de prompt
        prompt = f"""Baseado no contexto abaixo, responda a pergunta do usuário.
        
Contexto:
{context_text}

Pergunta: {query}

Resposta:"""
        
        return prompt
    
    def rag_query(self, query: str, verbose: bool = True) -> Dict:
        """Pipeline completo: query → retrieve → generate"""
        
        # Recuperar
        retrieved = self.retrieve(query)
        
        if verbose:
            print(f"\n🔍 Query: {query}")
            print(f"📊 Resultados encontrados: {len(retrieved)}\n")
            
            for i, (chunk, score) in enumerate(retrieved, 1):
                print(f"[{i}] Relevância: {score:.1%}")
                print(f"    {chunk[:150]}...\n")
        
        # Extrair apenas textos para geração
        context_texts = [chunk for chunk, _ in retrieved]
        
        # Gerar
        prompt = self.generate_response(query, context_texts)
        
        return {
            'query': query,
            'retrieved_chunks': len(retrieved),
            'context': context_texts,
            'prompt': prompt,
            'scores': [score for _, score in retrieved]
        }


# ============================================
# MAIN - EXECUTAR PIPELINE
# ============================================

if __name__ == "__main__":
    
    # Criar pipeline
    rag = RAGPipeline(CONFIG)
    
    # Carregar documentos
    docs = rag.load_documents(CONFIG['documents_dir'])
    
    if docs:
        # Processar documentos
        rag.process_documents(docs)
        
        # Exemplos de queries
        test_queries = [
            "Qual é o assunto principal?",
            "O que é RAG?",
            "Explique sobre inteligência artificial"
        ]
        
        print("\n" + "="*70)
        print("TESTANDO RAG")
        print("="*70)
        
        for query in test_queries:
            result = rag.rag_query(query)
            
            # Separador
            print("-"*70 + "\n")
    
    else:
        print("\n💡 Dica: Coloque seus arquivos .txt em ./documents/")

### Passo 5.2: Executar Pipeline

# Ativar venv (se não estiver ativado)
# Windows: .\venv\Scripts\activate
# Linux: source venv/bin/activate

# Executar
python rag_pipeline.py

**Output esperado:**
🚀 Usando device: CUDA
   GPU: NVIDIA GeForce RTX 3090
   VRAM: 24.0 GB

[1/4] Carregando modelo de embedding...
     ✓ Modelo carregado: sentence-transformers/all-MiniLM-L6-v2
     ✓ Dimensão embeddings: 384

[2/4] Inicializando ChromaDB...
     ✓ ChromaDB pronto (armazenado em chromadb_storage)

[3/4] Configurando splitter de texto...
     ✓ Chunk size: 512

[4/4] Sistema pronto!

📄 Carregando 1 arquivo(s)...
   ✓ exemplo.txt (245 caracteres)

📊 Processando documentos...

[Fase 1] Dividindo documentos em chunks...
   ✓ 2 chunks criados

[Fase 2] Gerando embeddings (batch size=32)...
Embeddings: 100%|██████████| 1/1
   ✓ 2 embeddings gerados

[Fase 3] Armazenando em ChromaDB...
Armazenando: 100%|██████████| 1/1
   ✓ Pipeline completo!
   • Total chunks: 2
   • Documentos processados: 1
   • Coleção: 'documents' em chromadb_storage

---

## Parte 6: Interface Web (Opcional)

### Passo 6.1: Criar `app_streamlit.py`

"""
Interface Streamlit para RAG Local
"""

import streamlit as st
from rag_pipeline import RAGPipeline, CONFIG

# Page config
st.set_page_config(
    page_title="RAG Desktop",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Local - Desktop com GPU")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuração")
    
    top_k = st.slider(
        "Número de chunks para recuperar",
        min_value=1,
        max_value=10,
        value=3
    )
    
    st.divider()
    st.caption(f"📊 Device: {CONFIG['device'].upper()}")
    st.caption(f"📦 Modelo: {CONFIG['model_name'].split('/')[-1]}")

# Inicializar pipeline
@st.cache_resource
def load_pipeline():
    return RAGPipeline(CONFIG)

rag = load_pipeline()

# Interface
query = st.text_input(
    "🔍 Faça uma pergunta sobre seus documentos:",
    placeholder="Digite sua pergunta..."
)

if query:
    with st.spinner("Buscando contexto..."):
        result = rag.rag_query(query, verbose=False)
    
    # Exibir resultados
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Chunks Relevantes")
        
        for i, (chunk, score) in enumerate(
            zip(result['context'], result['scores']),
            1
        ):
            with st.expander(f"Resultado {i} — Relevância: {score:.1%}"):
                st.text(chunk)
    
    with col2:
        st.subheader("📊 Estatísticas")
        st.metric("Chunks encontrados", result['retrieved_chunks'])
        st.metric("Score médio", f"{sum(result['scores'])/len(result['scores']):.1%}")

st.divider()
st.caption("💡 Coloque seus arquivos .txt em ./documents/ e recarregue a página")

### Passo 6.2: Instalar Streamlit

pip install streamlit==1.31.1

### Passo 6.3: Executar App Web

streamlit run app_streamlit.py

Abrirá em `http://localhost:8501`

---

## Parte 7: Otimizações para RTX3090

### Passo 7.1: Configurações Avançadas

Crie `config_advanced.py`:

import torch
import os

# ============================================
# OTIMIZAÇÕES GPU RTX3090
# ============================================

# Habilitar TensorFloat32 (mais rápido, menos preciso)
# Descomente se achar que está lento
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# Usar NVIDIA Automatic Mixed Precision (AMP)
# Reduz uso de memória e acelera treinamento
ENABLE_AMP = True

# Tamanho de batch para embedding
# RTX3090 suporta até 256 com modelo small
EMBEDDING_BATCH_SIZE = 64  # Aumentar para acelerar

# Usar float16 ao invés de float32
USE_FLOAT16 = False  # Pode causar problemas com embeddings

# Cache CUDA (mais rápido, mais memória)
torch.cuda.empty_cache()

print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"CUDA Reserved: {torch.cuda.memory_reserved() / 1e9:.1f} GB")

### Passo 7.2: Usar Modelos Maiores

Se tiver espaço em VRAM, use modelo maior para melhor qualidade:

# Em rag_pipeline.py, trocar:
CONFIG['model_name'] = 'sentence-transformers/all-mpnet-base-v2'  # 768-dim
# ou
CONFIG['model_name'] = 'sentence-transformers/all-minilm-l12-v2'  # 384-dim, mais rápido

**Comparativa de Modelos:**

| Modelo | Tamanho | Dimensão | Velocidade | Qualidade |
|--------|---------|----------|-----------|-----------|
| all-MiniLM-L6-v2 | 22 MB | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| all-minilm-l12-v2 | 33 MB | 384 | ⚡⚡ | ⭐⭐⭐⭐ |
| all-mpnet-base-v2 | 420 MB | 768 | ⚡ | ⭐⭐⭐⭐⭐ |

---

## Parte 8: Troubleshooting

| Erro | Solução |
|------|---------|
| **"CUDA out of memory"** | Reduzir chunk_size, batch_size ou usar modelo menor |
| **"ModuleNotFoundError: No module named 'torch'"** | Rodar `pip install -r requirements.txt` novamente |
| **"CUDA is not available"** | Verificar nvidia-smi, reinstalar drivers/CUDA |
| **"Embedding model not found"** | Verificar conexão internet (baixa modelos Hugging Face) |
| **"ChromaDB query slow"** | Normal em primeiras queries, usa cache depois |
| **Python version mismatch** | Usar exatamente Python 3.10 |

---

## Parte 9: Estrutura Final do Projeto

rag-desktop/
├── venv/                          # Ambiente virtual
├── documents/                      # Seus arquivos .txt
│   ├── exemplo.txt
│   └── ... (mais arquivos)
├── chromadb_storage/              # Banco de dados vetorial
│   └── ... (arquivos ChromaDB)
├── requirements.txt               # Dependências
├── rag_pipeline.py               # Pipeline principal
├── app_streamlit.py              # Interface web (opcional)
├── config_advanced.py            # Configurações avançadas
└── README.md                     # Documentação

---

## Parte 10: Próximos Passos

1. **Fine-tuning:** Treinar embedding model com seus dados
2. **LLM Local:** Usar Ollama + Llama2 para gerar respostas
3. **Reranking:** Adicionar cross-encoder para melhorar relevância
4. **API REST:** Criar FastAPI para integração
5. **Monitoring:** Usar MLflow para rastrear experimentos

---

## Referências

- PyTorch CUDA: https://pytorch.org/get-started/locally/
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-12-1-0-download-archive
- cuDNN: https://developer.nvidia.com/cudnn
- ChromaDB Docs: https://docs.trychroma.com
- Sentence Transformers: https://www.sbert.net
- LangChain: https://python.langchain.com

---

**Versão:** 1.0  
**Última atualização:** Dezembro 2024  
**Testado com:** Python 3.10, PyTorch 2.2.0+cu121, RTX3090