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
    
    # Botão para reprocessar documentos
    if st.button("🔄 Reprocessar Documentos"):
        st.cache_resource.clear()
        st.rerun()

# Inicializar pipeline
@st.cache_resource
def load_pipeline():
    rag = RAGPipeline(CONFIG)
    # Tentar carregar collection existente
    try:
        rag.collection = rag.client.get_collection(name="documents")
        st.sidebar.success("✓ Collection carregada")
    except:
        st.sidebar.warning("⚠️ Collection não encontrada. Execute o processamento primeiro.")
    return rag

rag = load_pipeline()

# Verificar se há collection
if rag.collection is None:
    st.warning("⚠️ Nenhum documento processado ainda!")
    st.info("Execute primeiro: `python rag_pipeline.py` para processar seus documentos.")
    st.stop()

# Verificar se há documentos na collection
try:
    count = rag.collection.count()
    st.sidebar.metric("📊 Chunks indexados", count)
    if count == 0:
        st.error("❌ Collection vazia! Execute: `python rag_pipeline.py`")
        st.stop()
except:
    st.error("❌ Erro ao acessar collection!")
    st.stop()

# Interface
query = st.text_input(
    "🔍 Faça uma pergunta sobre seus documentos:",
    placeholder="Digite sua pergunta..."
)

if query:
    with st.spinner("Buscando contexto..."):
        result = rag.rag_query(query, verbose=False, top_k=top_k)
    
    # Verificar se há resultados
    if not result['scores'] or len(result['scores']) == 0:
        st.warning("⚠️ Nenhum resultado encontrado para sua pergunta.")
        st.info("Tente reformular a pergunta ou adicionar mais documentos.")
    else:
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
            
            # Calcular média de forma segura
            if result['scores']:
                avg_score = sum(result['scores']) / len(result['scores'])
                st.metric("Score médio", f"{avg_score:.1%}")
            else:
                st.metric("Score médio", "N/A")

st.divider()
st.caption("💡 Coloque seus arquivos .txt em ./documents/ e execute `python rag_pipeline.py`")
