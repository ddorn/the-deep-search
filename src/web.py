import asyncio
from anyio import Path
from config import load_config
from constants import DIRS
from core_types import AssetType
from storage import DATABASES, get_db, set_db, Database
from strategies.embed_chunks import EmbedChunksStrategy

import streamlit as st

with st.sidebar:
    if st.button("Reload db"):
        st.cache_resource.clear()

    config_path = st.text_input("Config path", value="./data/config-simple.yaml")
    config = load_config(Path(config_path))
    st.write(config.model_dump())


DATABASES.clear()
set_db("default", Database(DIRS.user_data_path / "db.sqlite", config=config))
db = get_db()

embeddings, chunk_to_idx = db.load_embeddings()
idx_to_chunk = {v: k for k, v in chunk_to_idx.items()}

st.sidebar.write(f"Loaded {len(embeddings)} embeddings!")

@st.cache_data
def embed(text: str):
    embedding = asyncio.run(EmbedChunksStrategy(None).embed_texts([text]))[0]
    return embedding

query = st.text_input("Query", value="What is the capital of France?")

if query:
    embedding = embed(query)

    distances = embeddings @ embedding
    top5 = distances.argsort()[-5:][::-1]
    top_chunks_ids = [idx_to_chunk[i] for i in top5]
    top_chunks = db.get_chunks(top_chunks_ids)

    docs_ids = set(chunk.document_id for chunk in top_chunks)
    text_assets = [
        db.get_asset_for_document(doc_id, AssetType.TEXT_FILE)[0]
        for doc_id in docs_ids
    ]


    tabs = st.tabs([str(asset.path) for asset in text_assets])
    for tab, asset in zip(tabs, text_assets):
        path = asset.path
        with tab:
            st.markdown(f"### {path}")
            # For .md and .txt files, we can read the text
            if path.suffix in [".md", ".txt"]:
                st.markdown(path.read_text())
            else:
                st.code(path.read_text())