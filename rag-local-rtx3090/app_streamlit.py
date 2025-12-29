"""
Interface Streamlit para RAG Local com Detecção de Verbetes
"""

import streamlit as st
import re
from rag_com_serafim import RAGPipeline, CONFIG

# Page config
st.set_page_config(
    page_title="RAG DHBB - Desktop",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG DHBB - Busca em Verbetes Biográficos")

# ============================================
# FUNÇÕES AUXILIARES PARA NOMES
# ============================================
def normalize_name_for_search(name: str) -> list:
    """
    Normaliza um nome para busca, considerando diferentes formatos.
    Retorna lista de variações do nome para buscar.
    
    Ex: "Ulysses Guimarães" → ["ulysses guimarães", "guimarães, ulysses"]
    """
    name_lower = name.lower().strip()
    variations = [name_lower]
    
    # Se tiver múltiplas palavras, criar variação "SOBRENOME, Nome"
    parts = name_lower.split()
    if len(parts) >= 2:
        # "Ulysses Guimarães" → "guimarães, ulysses"
        sobrenome = parts[-1]
        nome = " ".join(parts[:-1])
        variations.append(f"{sobrenome}, {nome}")
    
    return variations


def match_person_name(person_name: str, search_term: str) -> float:
    """
    Verifica se person_name corresponde ao termo de busca.
    Retorna score de match (0.0 a 1.0), sendo 0 = sem match.
    """
    person_lower = person_name.lower()
    search_lower = search_term.lower().strip()
    
    # Extrair partes do nome na base (formato "SOBRENOME, Nome")
    if "," in person_name:
        parts = person_name.split(",", 1)
        sobrenome_base = parts[0].strip().lower()
        nome_base = parts[1].strip().lower()
        nome_completo_base = f"{nome_base} {sobrenome_base}"
    else:
        nome_completo_base = person_lower
        sobrenome_base = person_lower.split()[-1] if " " in person_lower else person_lower
        nome_base = " ".join(person_lower.split()[:-1]) if " " in person_lower else person_lower
    
    # Extrair partes do termo de busca
    search_parts = search_lower.split()
    
    # Score de match
    score = 0.0
    
    # Match exato completo = score máximo
    if search_lower == nome_completo_base or search_lower == person_lower:
        return 1.0
    
    # Match de todas as palavras do termo de busca no nome
    if all(part in nome_completo_base for part in search_parts):
        # Quanto mais palavras, maior o score
        score = 0.8 + (0.2 * len(search_parts) / (len(search_parts) + 1))
        return score
    
    # Match de todas as palavras em qualquer ordem
    if all(part in person_lower for part in search_parts):
        score = 0.6 + (0.2 * len(search_parts) / (len(search_parts) + 1))
        return score
    
    # Match parcial (pelo menos metade das palavras)
    matches = sum(1 for part in search_parts if part in nome_completo_base)
    if matches >= len(search_parts) / 2:
        score = 0.3 * (matches / len(search_parts))
        return score
    
    return 0.0


def search_person_by_partial_name(rag_pipeline: RAGPipeline, partial_name: str, min_score: float = 0.6) -> list:
    """
    Busca pessoas cujo nome corresponda ao termo fornecido.
    Retorna lista de tuplas (nome, score) ordenada por score.
    min_score: score mínimo para considerar um match (default 0.6)
    """
    try:
        sample = rag_pipeline.collection.get(limit=10000)
        if sample and sample['metadatas']:
            matches = []
            for metadata in sample['metadatas']:
                person_name = metadata.get('person_name', '')
                score = match_person_name(person_name, partial_name)
                if score >= min_score:
                    matches.append((person_name, score))
            
            # Ordenar por score decrescente
            matches.sort(key=lambda x: x[1], reverse=True)
            return [name for name, score in matches]
    except Exception as e:
        st.error(f"❌ Erro na busca: {str(e)}")
    return []


# ============================================
# FUNÇÃO PARA DETECTAR NOME NA PERGUNTA
# ============================================
def detect_person_name_in_query(query: str, rag_pipeline: RAGPipeline) -> str:
    """
    Tenta identificar o nome de uma pessoa na pergunta.
    Busca nomes próprios capitalizados e valida na base de dados.
    """
    # Padrão: Nomes próprios com 2+ palavras capitalizadas
    # Ex: "Quem foi Ulysses Guimarães?" → "Ulysses Guimarães"
    pattern = r'\b([A-ZÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ][a-zàáâãäçèéêëìíîïñòóôõöùúûüý]+(?:\s+(?:da|de|do|dos|das|e)\s+)?(?:\s+[A-ZÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ][a-zàáâãäçèéêëìíîïñòóôõöùúûüý]+)+)\b'
    matches = re.findall(pattern, query)
    
    if matches:
        # Pegar o nome mais longo encontrado
        potential_name = max(matches, key=len)
        
        # Verificar se existe na base de dados com score alto
        matching_names = search_person_by_partial_name(rag_pipeline, potential_name, min_score=0.8)
        if matching_names:
            return matching_names[0]  # Retorna o match com maior score
        
        return potential_name
    
    return None


# ============================================
# SIDEBAR - CONFIGURAÇÕES
# ============================================
with st.sidebar:
    st.header("⚙️ Configuração")
    
    top_k = st.slider(
        "Número de chunks para recuperar",
        min_value=1,
        max_value=15,
        value=5
    )
    
    # Opção de forçar busca por pessoa
    st.divider()
    st.subheader("🔍 Filtro Manual")
    
    manual_person_filter = st.text_input(
        "Buscar apenas verbete de:",
        placeholder="Ex: Ulysses Guimarães",
        help="Digite nome completo ou sobrenome"
    )
    
    # Slider para sensibilidade da busca
    search_sensitivity = st.slider(
        "Sensibilidade da busca",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Maior = mais restritivo, menor = mais permissivo"
    )
    
    st.divider()
    st.caption(f"📊 Device: {CONFIG['device'].upper()}")
    st.caption(f"📦 Modelo: {CONFIG['model_name'].split('/')[-1]}")
    
    # Botão para reprocessar documentos
    if st.button("🔄 Reprocessar Documentos"):
        st.cache_resource.clear()
        st.rerun()


# ============================================
# INICIALIZAR PIPELINE
# ============================================
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
    
    # Buscar pessoas únicas
    try:
        all_metadata = rag.collection.get(limit=10000)
        unique_persons = set()
        if all_metadata and all_metadata['metadatas']:
            unique_persons = {m['person_name'] for m in all_metadata['metadatas'] if 'person_name' in m}
        
        st.sidebar.metric("📊 Chunks indexados", count)
        st.sidebar.metric("👤 Pessoas indexadas", len(unique_persons))
    except:
        st.sidebar.metric("📊 Chunks indexados", count)
    
    if count == 0:
        st.error("❌ Collection vazia! Execute: `python rag_pipeline.py`")
        st.stop()
except:
    st.error("❌ Erro ao acessar collection!")
    st.stop()


# ============================================
# INTERFACE PRINCIPAL
# ============================================
query = st.text_input(
    "🔍 Faça uma pergunta sobre os verbetes biográficos:",
    placeholder="Ex: Quem foi Ulysses Guimarães? / Qual a trajetória de Tancredo Neves?"
)

if query:
    with st.spinner("🔎 Analisando pergunta e buscando contexto..."):
        # Detectar nome na pergunta
        detected_name = None
        filters = None
        filter_names = []
        
        # Prioridade 1: Filtro manual do usuário
        if manual_person_filter.strip():
            detected_name = manual_person_filter.strip()
            # Buscar nomes que correspondam
            filter_names = search_person_by_partial_name(rag, detected_name, min_score=search_sensitivity)
            
            if filter_names:
                if len(filter_names) == 1:
                    filters = {"person_name": {"$eq": filter_names[0]}}
                    st.info(f"🎯 Busca manual: Filtrando por **{filter_names[0]}**")
                else:
                    # Múltiplas correspondências
                    filters = {"person_name": {"$in": filter_names}}
                    st.info(f"🎯 Busca manual: Encontrados **{len(filter_names)}** verbetes")
                    with st.expander("Ver nomes encontrados"):
                        for name in filter_names:
                            st.caption(f"• {name}")
            else:
                st.warning(f"⚠️ Nenhum verbete encontrado para '{detected_name}' (sensibilidade {search_sensitivity:.1f})")
                st.info("💡 Tente diminuir a sensibilidade da busca na barra lateral")
        
        # Prioridade 2: Detecção automática
        else:
            detected_name = detect_person_name_in_query(query, rag)
            if detected_name:
                # Buscar com score alto para evitar falsos positivos
                filter_names = search_person_by_partial_name(rag, detected_name, min_score=0.8)
                
                if filter_names:
                    if len(filter_names) == 1:
                        filters = {"person_name": {"$eq": filter_names[0]}}
                        st.success(f"✨ Detecção automática: Encontrado verbete de **{filter_names[0]}**")
                    else:
                        filters = {"person_name": {"$in": filter_names}}
                        st.success(f"✨ Detecção automática: Encontrados **{len(filter_names)}** verbetes possíveis")
                        with st.expander("Ver nomes encontrados"):
                            for name in filter_names:
                                st.caption(f"• {name}")
                else:
                    st.info(f"🌐 Nome '{detected_name}' detectado, mas sem match forte na base. Buscando em todos os verbetes.")
            else:
                st.info("🌐 Busca ampla: Consultando todos os verbetes")
        
        # Executar busca
        result = rag.rag_query(query, verbose=False, top_k=top_k, filters=filters)
    
    # Verificar se há resultados
    if not result['scores'] or len(result['scores']) == 0:
        st.warning("⚠️ Nenhum resultado encontrado para sua pergunta.")
        if filters:
            st.info("💡 Tente remover o filtro ou diminuir a sensibilidade da busca.")
        else:
            st.info("💡 Tente reformular a pergunta ou adicionar mais palavras-chave.")
    else:
        # ============================================
        # EXIBIR RESULTADOS
        # ============================================
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("📊 Estatísticas")
            st.metric("🔢 Chunks encontrados", result['retrieved_chunks'])
            
            # Calcular média de forma segura
            if result['scores']:
                avg_score = sum(result['scores']) / len(result['scores'])
                st.metric("⭐ Score médio", f"{avg_score:.1%}")
            else:
                st.metric("⭐ Score médio", "N/A")
            
            # Pessoas únicas nos resultados
            persons_in_results = set()
            if result['metadata']:
                persons_in_results = {m.get('person_name', 'N/A') for m in result['metadata']}
            
            if len(persons_in_results) > 1:
                st.metric("👥 Pessoas mencionadas", len(persons_in_results))
                with st.expander("Ver nomes"):
                    for person in sorted(persons_in_results):
                        st.caption(f"• {person}")
        
        with col1:
            st.subheader("📚 Chunks Relevantes")
            
            # Exibir cada chunk com todos os metadados
            for i, (chunk, score, metadata) in enumerate(
                zip(result['context'], result['scores'], result['metadata']),
                1
            ):
                # Título do expander com informações principais
                person_name = metadata.get('person_name', 'N/A')
                chunk_info = metadata.get('chunk_id', '?')
                total_chunks = metadata.get('total_chunks', '?')
                source = metadata.get('source', 'N/A')
                
                with st.expander(
                    f"**Resultado {i}** — 👤 {person_name} — ⭐ {score:.1%}",
                    expanded=(i == 1)  # Primeiro resultado expandido por padrão
                ):
                    # Metadados em destaque
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    
                    with meta_col1:
                        st.caption("**📁 Arquivo:**")
                        st.text(source)
                    
                    with meta_col2:
                        st.caption("**🔢 Chunk:**")
                        st.text(f"{chunk_info + 1}/{total_chunks}")
                    
                    with meta_col3:
                        st.caption("**📏 Tamanho:**")
                        st.text(f"{metadata.get('char_count', len(chunk))} chars")
                    
                    # Metadados adicionais em uma linha
                    meta_extra = []
                    if metadata.get('natureza'):
                        meta_extra.append(f"🏛️ {metadata['natureza']}")
                    if metadata.get('sexo'):
                        sexo_label = "♂️ Masculino" if metadata['sexo'] == 'm' else "♀️ Feminino"
                        meta_extra.append(sexo_label)
                    if metadata.get('cargos'):
                        cargos_text = metadata['cargos']
                        if len(cargos_text) > 100:
                            cargos_text = cargos_text[:100] + "..."
                        meta_extra.append(f"💼 {cargos_text}")
                    
                    if meta_extra:
                        st.caption(" | ".join(meta_extra))
                    
                    st.divider()
                    
                    # Conteúdo do chunk
                    st.markdown("**📄 Conteúdo:**")
                    st.text_area(
                        label="Chunk",
                        value=chunk,
                        height=300,
                        key=f"chunk_{i}",
                        label_visibility="collapsed"
                    )
                    
                    # Botão para copiar
                    if st.button(f"📋 Copiar chunk {i}", key=f"copy_{i}"):
                        st.code(chunk, language=None)

st.divider()
st.caption("💡 Coloque seus arquivos .text em ./documents/ e execute `python rag_pipeline.py`")
st.caption("🔬 Sistema inteligente de match: 'Ulysses Guimarães' → 'GUIMARÃES, Ulysses'")
