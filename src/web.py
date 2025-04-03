import asyncio
from anyio import Path
from config import load_config
from constants import DIRS
from storage import DATABASES, get_db, set_db, Database
from strategies.embed_chunks import EmbedChunksStrategy

import streamlit as st

st.title("My search")

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


st.write(f"Loaded {len(embeddings)} embeddings!")
st.write(str(idx_to_chunk))

@st.cache_data
def embed(text: str):
    embedding = asyncio.run(EmbedChunksStrategy(None).embed_texts([text]))[0]
    return embedding

with st.form(key="query_form"):
    query = st.text_input("Query", value="What is the capital of France?")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    embedding = embed(query)

    distances = embeddings @ embedding
    top5 = distances.argsort()[-5:][::-1]
    st.write(top5)
    top_chunks_ids = [idx_to_chunk[i] for i in top5]
    top_chunks = db.get_chunks(top_chunks_ids)

    st.write("Top 5 chunks:")
    for i, chunk in enumerate(top_chunks):
        st.write(f"**{i}**: {chunk.content}")