import streamlit as st
import numpy as np
import ollama
import hnswlib
import fitz  # PyMuPDF
import pickle
from typing import List
import os

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    try:
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        total_pages = len(pdf_document)
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            text += page.get_text()
            text += "\n\n"
        
        pdf_document.close()
        
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    text = text.replace('\n\n\n', '\n\n')
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + para + "\n\n"
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama."""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return np.array(response['embedding'], dtype=np.float32)
    except Exception as e:
        return None

def create_embeddings(chunks: List[str], progress_bar, status_text) -> np.ndarray:
    """Create embeddings for all chunks."""
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Embedding chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        if embedding is not None:
            embeddings.append(embedding)
        progress_bar.progress((i + 1) / len(chunks))
    
    return np.array(embeddings, dtype=np.float32)

def create_hnsw_index(embeddings: np.ndarray, ef_construction: int = 200, M: int = 16) -> hnswlib.Index:
    """Create HNSW index for fast ANN search."""
    num_elements = len(embeddings)
    
    index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    index.add_items(embeddings, np.arange(num_elements))
    index.set_ef(50)
    
    return index

def save_data(chunks: List[str], hnsw_index: hnswlib.Index, base_filename: str, source_filenames: List[str]):
    """Save chunks and HNSW index."""
    os.makedirs("data", exist_ok=True)
    
    filepath = os.path.join("data", base_filename)
    
    hnsw_index.save_index(f"{filepath}_hnsw.bin")
    
    data = {
        'chunks': chunks,
        'embedding_dim': EMBEDDING_DIM,
        'model': EMBEDDING_MODEL,
        'num_chunks': len(chunks),
        'source_files': source_filenames  # Changed to list
    }
    
    with open(f"{filepath}_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    return filepath

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

def delete_document(base_filename: str):
    """Delete a saved document."""
    try:
        filepath = os.path.join("data", base_filename)
        if os.path.exists(f"{filepath}_hnsw.bin"):
            os.remove(f"{filepath}_hnsw.bin")
        if os.path.exists(f"{filepath}_data.pkl"):
            os.remove(f"{filepath}_data.pkl")
        return True
    except Exception as e:
        st.error(f"Error deleting: {e}")
        return False

def main():
    st.set_page_config(
        page_title="PDF Ingestion System", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("📄 PDF Ingestion & Embedding System")
    st.markdown("Process single or multiple PDFs and create vector embeddings for RAG")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload & Process PDFs")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents", 
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process and embed"
        )
        
        if uploaded_files:
            # Show uploaded files
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            with st.expander("📋 Uploaded Files", expanded=True):
                total_size = 0
                for file in uploaded_files:
                    file_size = file.size / (1024 * 1024)
                    total_size += file_size
                    st.text(f"📄 {file.name} ({file_size:.2f} MB)")
                st.info(f"**Total size:** {total_size:.2f} MB")
            
            # Processing mode
            st.markdown("### Processing Mode")
            process_mode = st.radio(
                "How would you like to process these PDFs?",
                ["Merge into one knowledge base", "Process separately"],
                help="Merge: Combine all PDFs into one searchable document\nSeparate: Create individual knowledge bases for each PDF"
            )
            
            # Settings
            with st.expander("⚙️ Processing Settings", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Chunking Settings**")
                    chunk_size = st.number_input(
                        "Chunk size:", 
                        100, 2000, CHUNK_SIZE, step=50,
                        help="Larger chunks = more context per chunk"
                    )
                    chunk_overlap = st.number_input(
                        "Overlap:", 
                        0, 500, CHUNK_OVERLAP, step=10,
                        help="Overlap helps maintain context continuity"
                    )
                
                with col_b:
                    st.markdown("**HNSW Parameters**")
                    ef_construction = st.slider(
                        "ef_construction:", 
                        100, 400, 200, step=50,
                        help="Higher = better accuracy but slower"
                    )
                    M = st.slider(
                        "M (connections):", 
                        8, 64, 16, step=4,
                        help="Number of connections per element"
                    )
            
            # Filename input
            if process_mode == "Merge into one knowledge base":
                base_filename = st.text_input(
                    "Save as:", 
                    value="merged_documents",
                    help="Name for the merged knowledge base"
                )
            else:
                st.info("Each PDF will be saved with its original filename")
                base_filename = None
            
            st.markdown("---")
            
            # Process button
            if st.button("🚀 Process & Embed Documents", type="primary", use_container_width=True):
                if process_mode == "Merge into one knowledge base":
                    # MERGE MODE
                    if not base_filename:
                        st.error("Please provide a filename")
                    else:
                        process_merged(uploaded_files, base_filename, chunk_size, chunk_overlap, ef_construction, M)
                else:
                    # SEPARATE MODE
                    process_separate(uploaded_files, chunk_size, chunk_overlap, ef_construction, M)
    
    with col2:
        st.header("📚 Saved Documents")
        
        saved_docs = list_saved_documents()
        
        if saved_docs:
            st.success(f"✅ {len(saved_docs)} document(s) saved")
            
            for doc in saved_docs:
                with st.expander(f"📄 {doc}"):
                    # Load metadata
                    try:
                        filepath = os.path.join("data", doc)
                        with open(f"{filepath}_data.pkl", 'rb') as f:
                            data = pickle.load(f)
                        
                        # Handle both old (source_file) and new (source_files) format
                        if 'source_files' in data:
                            st.markdown(f"**Sources:** {len(data['source_files'])} file(s)")
                            for source in data['source_files']:
                                st.markdown(f"  • {source}")
                        else:
                            st.markdown(f"**Source:** {data.get('source_file', 'Unknown')}")
                        
                        st.markdown(f"**Chunks:** {data['num_chunks']}")
                        st.markdown(f"**Model:** {data['model']}")
                        st.markdown(f"**Embedding Dim:** {data['embedding_dim']}")
                        
                        total_words = sum(len(chunk.split()) for chunk in data['chunks'])
                        st.markdown(f"**Total Words:** {total_words:,}")
                        
                        # File sizes
                        hnsw_size = os.path.getsize(f"{filepath}_hnsw.bin") / 1024
                        data_size = os.path.getsize(f"{filepath}_data.pkl") / 1024
                        st.markdown(f"**Size:** {hnsw_size + data_size:.1f} KB")
                        
                        if st.button(f"🗑️ Delete", key=f"del_{doc}", use_container_width=True):
                            if delete_document(doc):
                                st.success(f"Deleted {doc}")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error loading metadata: {e}")
        else:
            st.info("No documents saved yet")
        
        st.markdown("---")
        
        st.header("ℹ️ System Info")
        
        # Check Ollama
        try:
            test_embedding = get_embedding("test")
            if test_embedding is not None:
                st.success(f"✅ Ollama running")
                st.success(f"✅ {EMBEDDING_MODEL} available")
            else:
                st.error("❌ Ollama not responding")
        except Exception as e:
            st.error(f"❌ Ollama error: {e}")
        
        st.markdown(f"**Embedding Model:** {EMBEDDING_MODEL}")
        st.markdown(f"**Embedding Dimension:** {EMBEDDING_DIM}")

def process_merged(uploaded_files, base_filename, chunk_size, chunk_overlap, ef_construction, M):
    """Process multiple PDFs and merge into one knowledge base."""
    with st.spinner("Processing PDF documents..."):
        try:
            all_chunks = []
            source_filenames = []
            total_words = 0
            
            # Create progress tracking
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            # Extract text from all PDFs
            status_text.text("📖 Extracting text from all PDFs...")
            all_text = ""
            
            for idx, pdf_file in enumerate(uploaded_files):
                status_text.text(f"📖 Extracting: {pdf_file.name} ({idx+1}/{len(uploaded_files)})")
                content = extract_text_from_pdf(pdf_file)
                pdf_file.seek(0)  # Reset file pointer
                
                if content.strip():
                    all_text += f"\n\n--- Source: {pdf_file.name} ---\n\n"
                    all_text += content
                    source_filenames.append(pdf_file.name)
                    total_words += len(content.split())
                
                overall_progress.progress((idx + 1) / (len(uploaded_files) * 3))
            
            if not all_text.strip():
                st.error("❌ No text extracted from any PDF.")
                return
            
            st.success(f"✅ Extracted {total_words:,} words from {len(source_filenames)} PDF(s)")
            
            # Chunk
            status_text.text("📝 Chunking combined document...")
            chunks = chunk_text(all_text, chunk_size, chunk_overlap)
            st.success(f"✅ Created {len(chunks)} chunks")
            overall_progress.progress(len(uploaded_files) / (len(uploaded_files) * 3))
            
            # Embed
            status_text.text(f"🔮 Creating embeddings with {EMBEDDING_MODEL}...")
            progress_bar = st.progress(0)
            embeddings = create_embeddings(chunks, progress_bar, status_text)
            progress_bar.empty()
            
            if len(embeddings) > 0:
                st.success(f"✅ Created {len(embeddings)} embeddings")
                overall_progress.progress((len(uploaded_files) * 2) / (len(uploaded_files) * 3))
                
                # Build index
                status_text.text("🚀 Building HNSW index...")
                index = create_hnsw_index(embeddings, ef_construction, M)
                st.success("✅ HNSW index built")
                
                # Save
                status_text.text("💾 Saving data...")
                saved_path = save_data(chunks, index, base_filename, source_filenames)
                
                overall_progress.progress(1.0)
                status_text.empty()
                overall_progress.empty()
                
                # Summary
                st.balloons()
                st.success("🎉 Processing Complete!")
                
                st.markdown("### 📊 Summary")
                col_x, col_y, col_z = st.columns(3)
                col_x.metric("PDFs Processed", len(source_filenames))
                col_y.metric("Total Chunks", len(chunks))
                col_z.metric("Total Words", f"{total_words:,}")
                
                st.info(f"💡 Document ID: `{base_filename}` - Ready to use in Chat App!")
            else:
                st.error("❌ No embeddings created. Check Ollama setup.")
        except Exception as e:
            st.error(f"❌ Error processing PDFs: {e}")
            st.exception(e)

def process_separate(uploaded_files, chunk_size, chunk_overlap, ef_construction, M):
    """Process each PDF separately."""
    with st.spinner("Processing PDF documents..."):
        try:
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, pdf_file in enumerate(uploaded_files):
                base_filename = pdf_file.name.replace('.pdf', '').replace(' ', '_')
                
                status_text.text(f"📖 Processing: {pdf_file.name} ({idx+1}/{len(uploaded_files)})")
                
                # Extract
                content = extract_text_from_pdf(pdf_file)
                pdf_file.seek(0)
                
                if not content.strip():
                    st.warning(f"⚠️ No text in {pdf_file.name}, skipping...")
                    continue
                
                word_count = len(content.split())
                
                # Chunk
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                
                # Embed
                progress_bar = st.progress(0)
                embeddings = create_embeddings(chunks, progress_bar, status_text)
                progress_bar.empty()
                
                if len(embeddings) > 0:
                    # Build index
                    index = create_hnsw_index(embeddings, ef_construction, M)
                    
                    # Save
                    save_data(chunks, index, base_filename, [pdf_file.name])
                    
                    results.append({
                        'name': pdf_file.name,
                        'chunks': len(chunks),
                        'words': word_count
                    })
                
                overall_progress.progress((idx + 1) / len(uploaded_files))
            
            overall_progress.empty()
            status_text.empty()
            
            # Summary
            if results:
                st.balloons()
                st.success(f"🎉 Processed {len(results)} PDF(s) successfully!")
                
                st.markdown("### 📊 Summary")
                for result in results:
                    with st.expander(f"✅ {result['name']}"):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Chunks", result['chunks'])
                        col_b.metric("Words", f"{result['words']:,}")
                
                st.info("💡 All documents are ready to use in the Chat App!")
            else:
                st.error("❌ No PDFs were successfully processed.")
                
        except Exception as e:
            st.error(f"❌ Error processing PDFs: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()