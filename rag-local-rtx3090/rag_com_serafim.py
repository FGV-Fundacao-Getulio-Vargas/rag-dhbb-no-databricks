"""
RAG Pipeline Local para Desktop com GPU RTX3090
Otimizado para verbetes biográficos com embeddings Serafim PT-BR
"""

import os
import glob
import torch
import re
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm

# ============================================
# CONFIGURAÇÃO
# ============================================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 🆕 Modelos Serafim para Português Brasil (escolha um)
    # Opção 1: STS (Semantic Textual Similarity) - para similaridade semântica
    # 'model_name': 'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder',  # 100M, 768-dim
    # 'model_name': 'PORTULAN/serafim-335m-portuguese-pt-sentence-encoder',  # 335M, 768-dim
    # 'model_name': 'PORTULAN/serafim-900m-portuguese-pt-sentence-encoder',  # 900M, 1536-dim ⭐
    # Opção 2: IR (Information Retrieval) - MELHOR para RAG/busca
    # 'model_name': 'PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir',  # 100M, 768-dim
    'model_name': 'PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir',  # 900M, 1536-dim ⭐⭐⭐⭐⭐
        # nem acredito!
    # Alternativa: Modelo multilíngue com bom suporte a PT-BR
    # 'model_name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # 278M, 768-dim
    # 'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # 22MB, 384-dim o padrão usado pelo George
    
    'chunk_size': 1000,
    'chunk_overlap': 300,
    'top_k': 3,
    'embeddings_cache': 'chromadb_storage',
    'documents_dir': './documents'
}

print(f"🚀 Usando device: {CONFIG['device'].upper()}")
if CONFIG['device'] == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# FUNÇÕES DE EXTRAÇÃO DE METADADOS
# ============================================

def extract_person_name(content):
    """Extrai o nome da pessoa do cabeçalho do verbete"""
    try:
        match = re.search(r'^title:\s*(.+?)(?:\n|$)', content, re.MULTILINE)
        if match:
            full_name = match.group(1).strip()
            full_name = full_name.strip('"').strip("'")
            return full_name
        return "Nome não identificado"
    except Exception as e:
        print(f"  ⚠️ Erro ao extrair nome: {str(e)}")
        return "Nome não identificado"


def extract_metadata_from_yaml(content):
    """Extrai metadados completos do cabeçalho YAML"""
    metadata = {
        "person_name": "Nome não identificado",
        "natureza": None,
        "sexo": None,
        "cargos": []
    }
    
    try:
        yaml_match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL | re.MULTILINE)
        
        if yaml_match:
            yaml_content = yaml_match.group(1)
            
            title_match = re.search(r'title:\s*(.+?)(?:\n|$)', yaml_content)
            if title_match:
                metadata["person_name"] = title_match.group(1).strip().strip('"').strip("'")
            
            natureza_match = re.search(r'natureza:\s*(.+?)(?:\n|$)', yaml_content)
            if natureza_match:
                metadata["natureza"] = natureza_match.group(1).strip()
            
            sexo_match = re.search(r'sexo:\s*(.+?)(?:\n|$)', yaml_content)
            if sexo_match:
                metadata["sexo"] = sexo_match.group(1).strip()
            
            cargos_section = re.search(r'cargos:\s*\n((?:\s+-\s+.+\n?)+)', yaml_content)
            if cargos_section:
                cargos_text = cargos_section.group(1)
                metadata["cargos"] = [
                    line.strip().lstrip('- ').strip() 
                    for line in cargos_text.split('\n') 
                    if line.strip().startswith('-')
                ]
        
    except Exception as e:
        print(f"  ⚠️ Erro ao extrair metadados YAML: {str(e)}")
    
    return metadata


def load_text_files_as_chunks_splitted(folder_path, chunk_size=1000, chunk_overlap=200):
    """Carrega arquivos .text e divide em chunks otimizados"""
    docs = []
    text_files = glob.glob(os.path.join(folder_path, "*.text"))
    
    if not text_files:
        print(f"⚠️ Nenhum arquivo .text encontrado em {folder_path}")
        return docs
    
    print(f"✓ Encontrados {len(text_files)} arquivos .text\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for file_path in sorted(text_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_name = os.path.basename(file_path)
                
                yaml_metadata = extract_metadata_from_yaml(content)
                person_name = yaml_metadata["person_name"]
                
                chunks = text_splitter.split_text(content)
                
                print(f" ✓ {file_name}")
                print(f"   👤 Pessoa: {person_name}")
                print(f"   📄 Total: {len(content)} caracteres → {len(chunks)} chunks\n")
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "file_path": file_path,
                            "person_name": person_name,
                            "natureza": yaml_metadata["natureza"],
                            "sexo": yaml_metadata["sexo"],
                            "cargos": ", ".join(yaml_metadata["cargos"]) if yaml_metadata["cargos"] else None,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    docs.append(doc)
                    
        except Exception as e:
            print(f" ✗ Erro ao carregar {file_path}: {str(e)}")
    
    return docs


# ============================================
# CLASSE RAG PIPELINE
# ============================================
class RAGPipeline:
    def __init__(self, config: Dict):
        """Inicializar pipeline RAG com Serafim PT"""
        self.config = config
        self.device = config['device']
        
        print("\n[1/4] Carregando modelo de embedding Serafim PT...")
        print(f"   📦 Modelo: {config['model_name']}")
        
        try:
            self.embedding_model = SentenceTransformer(
                config['model_name'],
                device=self.device,
                trust_remote_code=True  # 🆕 Necessário para modelos Serafim
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"   ✓ Modelo carregado com sucesso!")
            print(f"   ✓ Dimensão embeddings: {self.embedding_dim}")
        except Exception as e:
            print(f"   ❌ Erro ao carregar modelo: {str(e)}")
            print(f"   💡 Tentando instalar dependências...")
            print(f"   💡 Execute: pip install sentence-transformers transformers torch")
            raise
        
        print("\n[2/4] Inicializando ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=config['embeddings_cache']
        )
        self.collection = None
        print(f"   ✓ ChromaDB pronto (armazenado em {config['embeddings_cache']})")
        
        print("\n[3/4] Configurando splitter de texto...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        print(f"   ✓ Chunk size: {config['chunk_size']}")
        print("\n[4/4] Sistema pronto!\n")

    def load_documents_with_metadata(self, directory: str) -> List[Document]:
        """Carregar documentos .text com metadados YAML extraídos"""
        print(f"📄 Carregando documentos de {directory}...")
        docs = load_text_files_as_chunks_splitted(
            directory, 
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        print(f"✅ {len(docs)} chunks carregados com metadados\n")
        return docs

    def process_documents_with_metadata(self, documents: List[Document]):
        """Processar documentos com metadados: embedding + armazenar"""
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
        
        # Fase 1: Gerar embeddings (em batches)
        print(f"[Fase 1] Gerando embeddings com Serafim PT (batch size=32)...")
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Embeddings"):
            batch_docs = documents[i:i+batch_size]
            texts = [doc.page_content for doc in batch_docs]
            
            # Gerar embeddings com GPU
            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=True  # 🆕 Normalização recomendada para IR
                )
                batch_embeddings = batch_embeddings.cpu().numpy().tolist()
                embeddings.extend(batch_embeddings)
        
        print(f"   ✓ {len(embeddings)} embeddings gerados")
        
        # Fase 2: Armazenar em ChromaDB
        print(f"\n[Fase 2] Armazenando em ChromaDB...")
        
        def clean_metadata(metadata: Dict) -> Dict:
            """Remove valores None dos metadados"""
            return {
                key: value for key, value in metadata.items() 
                if value is not None
            }
        
        ids = [f'chunk_{i:06d}' for i in range(len(documents))]
        documents_list = [doc.page_content for doc in documents]
        metadatas = [
            clean_metadata({
                'source': doc.metadata['source'],
                'file_path': doc.metadata['file_path'],
                'person_name': doc.metadata['person_name'],
                'natureza': doc.metadata.get('natureza') or '',
                'sexo': doc.metadata.get('sexo') or '',
                'cargos': doc.metadata.get('cargos') or '',
                'chunk_id': doc.metadata['chunk_id'],
                'total_chunks': doc.metadata['total_chunks'],
                'char_count': len(doc.page_content)
            })
            for doc in documents
        ]
        
        # Adicionar em batches
        for i in tqdm(range(0, len(ids), batch_size), desc="Armazenando"):
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
        print(f"   • Total chunks: {len(documents)}")
        print(f"   • Pessoas diferentes: {len(set(doc.metadata['person_name'] for doc in documents))}")
        print(f"   • Coleção: 'documents' em {self.config['embeddings_cache']}")

    def retrieve(self, query: str, top_k: int = None, filters: Dict = None) -> List[Tuple[str, float, Dict]]:
        """Buscar chunks mais relevantes com metadados"""
        if top_k is None:
            top_k = self.config['top_k']
        
        if self.collection is None:
            print("❌ Nenhuma coleção carregada. Execute process_documents_with_metadata() primeiro.")
            return []
        
        # Gerar embedding da query com normalização
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True  # 🆕 Normalização para consistência
        ).cpu().numpy().tolist()
        
        # Buscar no ChromaDB (com filtros opcionais)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters if filters else None
        )
        
        retrieved = []
        if results['documents'] and results['documents'][0]:
            for doc, distance, metadata in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                # ChromaDB retorna distância (menor = mais similar)
                similarity = 1 / (1 + distance)
                retrieved.append((doc, similarity, metadata))
        
        return retrieved

    def generate_response(self, query: str, context: List[str]) -> str:
        """Gerar resposta com contexto (versão simples)"""
        if not context:
            return "Desculpe, não encontrei informações relevantes."
        
        context_text = "\n\n".join(context[:3])
        
        prompt = f"""Baseado no contexto abaixo, responda a pergunta do usuário.

Contexto:
{context_text}

Pergunta: {query}

Resposta:"""
        
        return prompt

    def rag_query(self, query: str, verbose: bool = True, top_k: int = None, filters: Dict = None) -> Dict:
        """Pipeline completo: query → retrieve → generate"""
        retrieved = self.retrieve(query, top_k=top_k, filters=filters)
        
        if verbose:
            print(f"\n🔍 Query: {query}")
            if filters:
                print(f"🔧 Filtros: {filters}")
            print(f"📊 Resultados encontrados: {len(retrieved)}\n")
            
            for i, (chunk, score, metadata) in enumerate(retrieved, 1):
                print(f"[{i}] Relevância: {score:.1%}")
                print(f"   👤 Pessoa: {metadata.get('person_name', 'N/A')}")
                print(f"   📁 Arquivo: {metadata.get('source', 'N/A')}")
                print(f"   🔢 Chunk: {metadata.get('chunk_id', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
                print(f"   {chunk[:150]}...\n")
        
        context_texts = [chunk for chunk, _, _ in retrieved]
        prompt = self.generate_response(query, context_texts)
        
        return {
            'query': query,
            'retrieved_chunks': len(retrieved),
            'context': context_texts,
            'metadata': [meta for _, _, meta in retrieved],
            'prompt': prompt,
            'scores': [score for _, score, _ in retrieved]
        }


# ============================================
# MAIN - EXECUTAR PIPELINE
# ============================================
if __name__ == "__main__":
    # Criar pipeline
    rag = RAGPipeline(CONFIG)
    
    # Carregar documentos com metadados
    docs = rag.load_documents_with_metadata(CONFIG['documents_dir'])
    
    if docs:
        # Processar documentos
        rag.process_documents_with_metadata(docs)
        
        # Exemplos de queries
        test_queries = [
            "Qual foi a missão de Abbink?",
            "Quem foi Ulysses Guimarães?",
            "Explique sobre a vice-governadora do DF"
        ]
        
        print("\n" + "="*70)
        print("TESTANDO RAG COM SERAFIM PT")
        print("="*70)
        
        for query in test_queries:
            result = rag.rag_query(query)
            print("-"*70 + "\n")
        
        # Exemplo de busca com filtros
        print("\n" + "="*70)
        print("TESTANDO BUSCA COM FILTROS")
        print("="*70)
        
        result = rag.rag_query(
            "Quais foram os principais cargos?",
            filters={"sexo": "f"}
        )
        
    else:
        print("\n💡 Dica: Coloque seus arquivos .text em ./documents/")
