import streamlit as st
import pandas as pd
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="RAG Brain Visualizer", layout="wide")

st.title("Vector Database Inspector")

# Connect to Chroma
CHROMA_PATH = os.getenv("CHROMA_PATH")
try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("rag_collection")
except:
    st.error("Collection not found. Run the pipeline first!")
    st.stop()

# Stats
count = collection.count()
st.metric("Total Chunks in DB", count)

# Search
st.divider()
search_query = st.text_input("Filter by Text", placeholder="Type a concept...")

# Data Viewer
st.subheader("Database Contents")

result = collection.get(include=["metadatas", "documents", "embeddings"]) 
ids = result['ids']
docs = result['documents']
metas = result['metadatas']
embeds = result['embeddings']

if ids:
    df_data = []
    for i, _id in enumerate(ids):
        row = {"ID": _id, "Content": docs[i]}
        row.update(metas[i])
        
        vector_preview = embeds[i][:5] 
        row["Vector (Preview)"] = str([round(x, 3) for x in vector_preview]) + "..."
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Filter
    if search_query:
         df = df[df['Content'].str.contains(search_query, case=False) | df['type'].str.contains(search_query, case=False)]

    # Display
    st.dataframe(df, use_container_width=True)
    
    # Detail Inspector
    st.divider()
    st.subheader("Micro-Inspector")
    selected_id = st.selectbox("Select Chunk ID to Inspect", df['ID'].unique())
    
    if selected_id:
        idx = ids.index(selected_id)
        row = df[df['ID'] == selected_id].iloc[0]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Type:** `{row.get('type', 'N/A')}`")
            st.markdown(f"**File ID:** `{row.get('file_id', 'N/A')}`")
            st.text_area("Full Content", row['Content'], height=300)
        with c2:
            st.markdown("**Full Embedding Vector:**")
            st.code(str(embeds[idx]), language="python")

else:
    st.info("Database is empty.")