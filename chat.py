import streamlit as st
import numpy as np
import ollama
import faiss
import pickle
from typing import List, Tuple
import os

# Configuration - Improved defaults
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "gemma3:latest"
EMBEDDING_DIM = 768
TEMPERATURE = 0.7
MAX_TOKENS = 2000
TOP_K_CHUNKS = 20  # ✅ Increased from 5 to 20
SIMILARITY_THRESHOLD = 0.3  # ✅ Filter out irrelevant chunks
FAISS_NPROBE = 50  # ✅ Number of clusters to search (for IVF indexes)

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama."""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embedding = np.array(response['embedding'], dtype=np.float32)
        # ✅ Normalize for cosine similarity with FAISS
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding
    except Exception as e:
        return None

def load_data(base_filename: str) -> Tuple[List[str], faiss.Index, dict]:
    """Load chunks and FAISS index."""
    filepath = os.path.join("data", base_filename)
    
    with open(f"{filepath}_data.pkl", 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    
    # ✅ Load FAISS index
    index = faiss.read_index(f"{filepath}_faiss.bin")
    
    # ✅ Set search parameters for IVF indexes
    if hasattr(index, 'nprobe'):
        index.nprobe = FAISS_NPROBE
    
    return chunks, index, data

def list_saved_documents():
    """List all saved documents in data directory."""
    if not os.path.exists("data"):
        return []
    
    files = os.listdir("data")
    docs = set()
    for f in files:
        if f.endswith("_data.pkl"):
            docs.add(f.replace("_data.pkl", ""))
    return sorted(list(docs))

def expand_query(query: str) -> List[str]:
    """✅ Expand query with variations for better retrieval."""
    queries = [query]
    
    # Add a rephrased version for better matching
    if len(query.split()) > 3:
        queries.append(f"Information about {query}")
        queries.append(f"Details regarding {query}")
    
    return queries

def search_similar_chunks(
    query: str, 
    faiss_index: faiss.Index, 
    chunks: List[str], 
    top_k: int = TOP_K_CHUNKS,
    threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[str, float, int]]:
    """✅ IMPROVED: Search with FAISS using multiple queries and relevance filtering."""
    
    # Expand query for better retrieval
    queries = expand_query(query)
    
    all_results = {}  # Use dict to deduplicate by chunk index
    
    for q in queries:
        query_embedding = get_embedding(q)
        if query_embedding is None:
            continue
        
        # ✅ FAISS search
        # Search for more candidates
        search_k = min(top_k * 2, len(chunks))
        query_vector = query_embedding.reshape(1, -1)
        
        # ✅ FAISS returns distances and indices
        # For normalized vectors with IndexFlatIP, distance = cosine similarity
        distances, indices = faiss_index.search(query_vector, search_k)
        
        for idx, distance in zip(indices[0], distances[0]):
            # ✅ For IndexFlatIP with normalized vectors, distance is already similarity (0-1)
            # For L2 distance, convert: similarity = 1 / (1 + distance)
            similarity = float(distance)  # Assuming IndexFlatIP
            
            # ✅ Filter by threshold and keep best score
            if similarity >= threshold and 0 <= idx < len(chunks):
                if idx not in all_results or similarity > all_results[idx][1]:
                    all_results[idx] = (chunks[idx], similarity, int(idx))
    
    # Sort by similarity and take top_k
    results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return results

def is_general_conversation(query: str) -> bool:
    """Detect if query is general conversation vs document question."""
    query_lower = query.lower().strip()
    
    general_patterns = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'whats up', "what's up", 'sup',
        'thanks', 'thank you', 'bye', 'goodbye',
        'who are you', 'what are you', 'what can you do',
        'help', 'ok', 'okay', 'cool', 'nice', 'great'
    ]
    
    for pattern in general_patterns:
        if query_lower == pattern or query_lower.startswith(pattern + ' '):
            return True
    
    if len(query.split()) <= 2 and '?' not in query:
        return True
    
    return False

def stream_general_answer(query: str):
    """Generate answer for general conversation without document context."""
    prompt = f"""You are a friendly AI assistant helping users chat with their documents.

User message: {query}

Instructions:
- Respond naturally and conversationally
- Be helpful and friendly
- If asked what you can do, explain you can answer questions about the loaded document
- Keep responses concise for greetings and casual chat

Response:"""
    
    try:
        stream = ollama.generate(
            model=DEFAULT_LLM_MODEL,
            prompt=prompt,
            stream=True,
            options={
                'temperature': 0.8,
                'num_predict': 200,
            }
        )
        for chunk in stream:
            yield chunk['response']
    except Exception as e:
        yield f"Error: {e}"

def stream_answer(query: str, context_chunks: List[Tuple[str, float, int]]):
    """✅ IMPROVED: Generate detailed answer with more context."""
    
    # Build context from chunks
    context_parts = []
    for i, (chunk, score, idx) in enumerate(context_chunks, 1):
        context_parts.append(f"[Section {i}]:\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful AI assistant with knowledge from a document. Answer the user's question naturally and conversationally using the provided context.

Context information from the document:
{context}

Question: {query}

Instructions:
- Provide a comprehensive answer using ALL relevant information from the context sections
- Synthesize information from multiple sections when needed
- Answer naturally without mentioning "the document", "the context", or "Section X"
- Speak as if you have direct knowledge of the subject
- Be thorough and detailed
- Include specific examples, data, and details from the context
- If the context doesn't contain enough information, acknowledge this naturally
- Organize your response clearly

Answer:"""
    
    try:
        stream = ollama.generate(
            model=DEFAULT_LLM_MODEL,
            prompt=prompt,
            stream=True,
            options={
                'temperature': TEMPERATURE,
                'num_predict': MAX_TOKENS,
                'top_p': 0.9,
                'top_k': 40,
            }
        )
        for chunk in stream:
            yield chunk['response']
    except Exception as e:
        yield f"Error generating answer: {e}"

def main():
    st.set_page_config(
        page_title="Chat with PDF", 
        page_icon="💬",
        layout="centered"
    )
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False
    
    # Get available documents
    saved_docs = list_saved_documents()
    
    # Auto-load first document if available and nothing is loaded
    if saved_docs and not st.session_state.data_loaded:
        try:
            with st.spinner("Loading document..."):
                first_doc = saved_docs[0]
                chunks, index, metadata = load_data(first_doc)
                st.session_state.chunks = chunks
                st.session_state.faiss_index = index
                st.session_state.metadata = metadata
                st.session_state.data_loaded = True
                st.session_state.current_doc = first_doc
        except Exception as e:
            st.error(f"Error auto-loading document: {e}")
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Documents")
        
        if saved_docs:
            selected_doc = st.selectbox(
                "Current document:",
                saved_docs,
                index=saved_docs.index(st.session_state.current_doc) if st.session_state.current_doc in saved_docs else 0,
                label_visibility="visible"
            )
            
            if selected_doc != st.session_state.current_doc and st.session_state.data_loaded:
                try:
                    with st.spinner("Loading..."):
                        chunks, index, metadata = load_data(selected_doc)
                        st.session_state.chunks = chunks
                        st.session_state.faiss_index = index
                        st.session_state.metadata = metadata
                        st.session_state.current_doc = selected_doc
                        st.session_state.messages = []
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            if st.session_state.data_loaded:
                st.markdown("---")
                st.caption(f"📄 {st.session_state.metadata.get('source_file', 'Unknown')}")
                st.caption(f"📊 {len(st.session_state.chunks)} chunks")
                
                # ✅ Retrieval settings
                st.markdown("---")
                st.subheader("⚙️ Settings")
                
                top_k = st.slider(
                    "Chunks to retrieve",
                    min_value=5,
                    max_value=50,
                    value=TOP_K_CHUNKS,
                    help="More chunks = better coverage but slower"
                )
                st.session_state.top_k = top_k
                
                threshold = st.slider(
                    "Similarity threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=SIMILARITY_THRESHOLD,
                    step=0.05,
                    help="Filter out irrelevant chunks"
                )
                st.session_state.threshold = threshold
                
                st.session_state.show_sources = st.checkbox(
                    "Show source chunks",
                    value=False,
                    help="Display which chunks were used"
                )
        else:
            st.warning("No documents found")
            st.info("Process PDFs in the Ingestion App first")
        
        st.markdown("---")
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    if st.session_state.data_loaded:
        st.title("💬 Chat")
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # ✅ Show sources if enabled
                    if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
                        with st.expander(f"📚 Sources ({len(message['sources'])} chunks used)"):
                            for i, (chunk, score, idx) in enumerate(message['sources'], 1):
                                st.markdown(f"**Chunk {idx}** (similarity: {score:.2f})")
                                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.markdown("---")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                sources = []
                
                if is_general_conversation(prompt):
                    for chunk in stream_general_answer(prompt):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                else:
                    # ✅ Use configurable parameters
                    top_k = st.session_state.get('top_k', TOP_K_CHUNKS)
                    threshold = st.session_state.get('threshold', SIMILARITY_THRESHOLD)
                    
                    with st.spinner(f"Searching {len(st.session_state.chunks)} chunks with FAISS..."):
                        relevant_chunks = search_similar_chunks(
                            prompt,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            top_k=top_k,
                            threshold=threshold
                        )
                        
                        sources = relevant_chunks
                        
                        if relevant_chunks:
                            # ✅ Show retrieval stats
                            avg_score = sum(score for _, score, _ in relevant_chunks) / len(relevant_chunks)
                            st.caption(f"Found {len(relevant_chunks)} relevant chunks (avg similarity: {avg_score:.2f})")
                            
                            for chunk in stream_answer(prompt, relevant_chunks):
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                        else:
                            full_response = f"I couldn't find relevant information (searched {len(st.session_state.chunks)} chunks). Try lowering the similarity threshold or rephrasing your question."
                            response_placeholder.markdown(full_response)
                
                response_placeholder.markdown(full_response)
                
                # ✅ Save with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
    
    else:
        st.title("💬 Chat with PDF")
        st.markdown("---")
        st.info("No documents found. Please process PDFs in the Ingestion App first.")
        st.markdown("""
        ### How it works
        
        1. Process a PDF in the Ingestion App
        2. Document will auto-load here
        3. Start asking questions in the chat
        
        The AI will answer based on the document's content.
        """)

if __name__ == "__main__":
    main()