import asyncio
from collections import defaultdict

import streamlit as st
from anyio import Path
from millify import millify

from config import load_config
from constants import DIRS, SYNC_PATTERN
from core_types import Asset, AssetType
from storage import DATABASES, Database, get_db, set_db
from strategies.embed_chunks import EmbedChunksStrategy

st.set_page_config(
    page_title="The deep search",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with st.sidebar:
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


text = st.text_area("Document", "no text")
nice_text = SYNC_PATTERN.sub("", text)
# Highlight the sync pattern
text = SYNC_PATTERN.sub(lambda x: f":blue[{x.group()}]", text)

st.markdown(text)
st.markdown(nice_text)


query = st.text_input("Query", value="delete")
nb_results = st.slider("Number of results", 1, 100, 5)

if query:
    embedding = embed(query)

    distances = embeddings @ embedding
    top_n = distances.argsort()[-nb_results:][::-1]
    top_chunks_ids = [idx_to_chunk[i] for i in top_n]
    top_chunks = db.get_chunks(top_chunks_ids)
    chunks = {chunk.id: chunk for chunk in top_chunks}

    if len(chunks) == 0:
        st.text("No chunks found")
        st.stop()

    docs_ids = set(chunk.document_id for chunk in top_chunks)

    documents = {doc_id: db.get_document(doc_id) for doc_id in docs_ids}
    assets: dict[int, dict[str, list[Asset]]] = {}
    for doc_id in docs_ids:
        assets[doc_id] = defaultdict(list)
        for asset in db.get_assets_for_document(doc_id):
            assets[doc_id][asset.type].append(asset)

    selected_chunk = st.session_state.get("selected_chunk", top_chunks[0].id)
    if selected_chunk not in chunks:
        selected_chunk = top_chunks[0].id
    result_col, doc_col = st.columns([1, 1])

    with result_col:
        st.header("Top results")
        for chunk, score in zip(top_chunks, distances[top_n]):
            doc = documents[chunk.document_id]
            cols = st.columns([8, 1])
            cols[0].markdown(
                f"### *{doc.source_id}* :blue[{doc.title}]\n Chunk: {chunk.document_order} -- Score: {score:.3f}"
            )
            if cols[1].button("👉", key=f"chunk_{chunk.id}", use_container_width=True):
                selected_chunk = chunk.id
                st.session_state.selected_chunk = selected_chunk

            st.code(chunk.content)

    with doc_col:
        chunk = chunks[selected_chunk]
        doc_assets = assets[chunk.document_id]
        doc = documents[chunk.document_id]

        # sync_ids = SYNC_PATTERN.findall(chunk.content)
        # st.write(sync_ids)
        if doc.url is not None:
            st.link_button(
                label="Open",
                url=doc.url,
                type="secondary",
                icon=":material/open_in_new:",
            )

        # pick the type to display
        type_ = st.radio("Type to show", list(doc_assets.keys()), index=0, horizontal=True)
        assets_of_type = doc_assets[type_]

        for asset in assets_of_type:
            if asset.path is not None:
                # For .md and .txt files, we can use .markdown, for others we can use .code
                if asset.path.suffix in [".md", ".txt"]:
                    st.markdown(f"### {asset.path}")
                    st.markdown(asset.path.read_text())
                else:
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


st.header("Statistics")

tabs = st.tabs(["Tasks", "Chunks"])

with tabs[0]:
    # Tasks per strategy (pending/in_progress/done)
    db.cursor.execute(
        """
        SELECT COUNT(*), strategy, status
        FROM tasks
        GROUP BY strategy, status
    """
    )
    rows = db.cursor.fetchall()
    st.write("### Number of tasks per strategy and status")

    # {strategy: {status: count}}
    stats = defaultdict(lambda: defaultdict(int))
    for row in rows:
        stats[row[1]][row[2]] += row[0]

    text = ""
    for strategy, status_counts in stats.items():
        # - strategy (red-pending, green-done, blue-in_progress)
        text += f"- {strategy} (:red[{status_counts['pending']}], :green[{status_counts['done']}], :blue[{status_counts['in_progress']}])\n"
    st.markdown(text)

# Show numbers of chunks per document
with tabs[1]:
    # Number of words in all chunks
    db.cursor.execute("SELECT content FROM chunks")
    word_count = sum(len(chunk["content"].split()) for chunk in db.cursor.fetchall())
    st.metric("Number of words in all chunks", millify(word_count, precision=1))

    # Number of characters in all chunks
    db.cursor.execute("SELECT SUM(LENGTH(content)) FROM chunks")
    rows = db.cursor.fetchall()
    st.metric("Number of characters in all chunks", millify(rows[0][0], precision=1))

    db.cursor.execute(
        """
        SELECT COUNT(*), document_id
        FROM chunks
        GROUP BY document_id
        """
    )
    rows = db.cursor.fetchall()
    st.write("### Number of chunks per document")
    stats = [{"document_id": row[1], "count": row[0]} for row in rows]
    stats.sort(key=lambda x: x["count"], reverse=True)
    for stat in stats:
        doc = db.get_document(stat["document_id"])
        st.markdown(f"Document {doc.source_id} ({doc.title}): {stat['count']} chunks")
