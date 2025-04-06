import asyncio
from collections import defaultdict
from anyio import Path
from config import load_config
from constants import DIRS
from core_types import Asset, AssetType
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

query = st.text_input("Query", value="delete")

if query:
    embedding = embed(query)

    distances = embeddings @ embedding
    top5 = distances.argsort()[-5:][::-1]
    top_chunks_ids = [idx_to_chunk[i] for i in top5]
    top_chunks = db.get_chunks(top_chunks_ids)

    docs_ids = set(chunk.document_id for chunk in top_chunks)

    assets: dict[int, dict[str, list[Asset]]] = {}
    for doc_id in docs_ids:
        assets[doc_id] = defaultdict(list)
        for asset in db.get_assets_for_document(doc_id):
            assets[doc_id][asset.type].append(asset)

    tabs = st.tabs([str(doc_id) for doc_id in docs_ids])
    for tab, doc_id in zip(tabs, docs_ids):
        with tab:
            doc_assets = assets[doc_id]
            # pick the type to display
            type_ = st.selectbox("Type", list(doc_assets.keys()), index=0, key=f"type_{doc_id}")
            assets_of_type = doc_assets[type_]

            for asset in assets_of_type:
                if asset.path is not None:
                    st.markdown(f"### {asset.path}")
                    st.code(asset.path.read_text())
                elif asset.type == AssetType.CHUNK_ID:
                    chunk = db.get_chunks([int(asset.content)])[0]
                    st.markdown(f"#### Chunk {chunk.document_order}")
                    st.code(chunk.content)
                else:
                    st.markdown(f"### Asset {asset.id}")
                    st.code(asset.content)

            # path = asset.path
            # st.markdown(f"### {path}")
            # # For .md and .txt files, we can read the text
            # if path.suffix in [".md", ".txt"]:
            #     st.markdown(path.read_text())
            # else:
            #     st.code(path.read_text())