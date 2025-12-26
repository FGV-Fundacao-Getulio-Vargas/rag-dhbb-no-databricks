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
        """Carregar todos os arquivos .text de um diretório"""
        documents = {}
        
        txt_files = glob.glob(os.path.join(directory, "*.text"))
        
        if not txt_files:
            print(f"⚠️  Nenhum arquivo .text encontrado em {directory}")
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
            "Qual foi a missão de Abbink?",
            "Explique sobre a vice-governadora do DF"
        ]
        
        print("\n" + "="*70)
        print("TESTANDO RAG")
        print("="*70)
        
        for query in test_queries:
            result = rag.rag_query(query)
            
            # Separador
            print("-"*70 + "\n")
    
    else:
        print("\n💡 Dica: Coloque seus arquivos .text em ./documents/")