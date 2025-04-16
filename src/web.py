import asyncio
import json
import re
from collections import defaultdict

import streamlit as st
import streamlit.components.v1 as components
from anyio import Path
from litellm import batch_completion
from millify import millify

from config import load_config
from constants import DIRS, SYNC_PATTERN
from core_types import Asset, AssetType
from storage import DATABASES, Database, set_db
from strategies.embed_chunks import EmbedChunksStrategy

st.set_page_config(
    page_title="The Deep Search",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

mark_linker = components.declare_component(
    "mark_linker",
    path="src/streamlit-components/mark_linker/mark_linker/frontend/build",
)

# Load config
config_path = st.sidebar.text_input("Config path", value="./data/config-simple.yaml")
config = load_config(Path(config_path))

# Load database
DATABASES.clear()
db = set_db("default", Database(DIRS.user_data_path / "db.sqlite", config=config))

# Load embeddings
embeddings, chunk_to_idx = db.load_embeddings()
idx_to_chunk = {v: k for k, v in chunk_to_idx.items()}


@st.cache_data
def embed(text: str):
    embedding = asyncio.run(EmbedChunksStrategy(None).embed_texts([text]))[0]
    return embedding


with st.sidebar:
    st.write(f"Loaded {len(embeddings)} embeddings.")
    st.write(config.model_dump())

query = st.text_input("Query", value="delete")
nb_results = st.slider("Number of results", 1, 100, 5)
debugging = st.checkbox("Debug mode", value=True)


def show_results(top_chunks):
    st.header("Top results")

    doc_to_chunks = defaultdict(list)

    # Group by document_id
    for chunk in top_chunks:
        doc_to_chunks[chunk.document_id].append(chunk)

    # Sort by document_order for each group
    for doc_id in doc_to_chunks:
        doc_to_chunks[doc_id].sort(key=lambda chunk: chunk.document_order)

    for doc_id, chunks in doc_to_chunks.items():
        doc = documents[doc_id]

        title_col, button_col = st.columns([8, 1])
        title_col.markdown(f"### *{doc.source_id}* {doc.title}")

        PROMPT = f"""
            Extract the most relevant excerpt (10-20 words) from the chunk of text with respect to the below query.

QUERY: {query}

Your response should:
- Contain only the most relevant excerpt from the text
- Be the most related to the above query
- Include at least 10 words.
- Include no additional text, introductions, or explanations
- Preserve the exact wording from the original text
"""

        responses = batch_completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[
                [
                    dict(role="system", content=PROMPT),
                    dict(role="user", content=f"CHUNK: {chunk.content}\n"),
                ]
                for chunk in chunks
            ],
        )

        for chunk, response in zip(chunks, responses):
            st.write(chunk.content)
            st.write(response.choices[0].message.content)


def show_selected_chunk(selected_chunk, assets, documents, chunks):
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
        asset_content = asset.content if asset.content is not None else asset.path.read_text()

        if asset.type == AssetType.SYNCED_TEXT_FILE:
            st.markdown(f"### {asset.path}")

            mark = re.search(SYNC_PATTERN, chunk.content).group("id")
            mark_linker(markdown=asset_content, highlighted_mark=mark)

        elif asset.type == AssetType.STRUCTURE:
            st.markdown(f"### {asset.path}")
            st.json(json.loads(asset_content))

        elif asset.path is not None:
            # For .md and .txt files, we can use .markdown, for others we can use .code
            if asset.path.suffix in [".md", ".txt"]:
                st.markdown(f"### {asset.path}")
                st.markdown(asset_content)
            else:
                st.markdown(f"### {asset.path}")
                st.code(asset_content)

        elif asset.type == AssetType.CHUNK_ID:
            chunk = db.get_chunks([int(asset_content)])[0]
            st.markdown(f"#### Chunk {chunk.document_order}")
            st.code(chunk.content)
        else:
            st.markdown(f"### Asset {asset.id}")
            st.code(asset_content)

            # path = asset.path
            # st.markdown(f"### {path}")
            # # For .md and .txt files, we can read the text
            # if path.suffix in [".md", ".txt"]:
            #     st.markdown(asset_content)
            # else:
            #     st.code(asset_content)


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

    st.write(
        f"Found **{len(chunks)} chunks** associated with **{len(documents)} documents** and **{len(assets)} assets**"
    )

    selected_chunk = st.session_state.get("selected_chunk", top_chunks[0].id)
    # If the selected chunk from the last query does not exist, we take a new one.
    if selected_chunk not in chunks:
        selected_chunk = top_chunks[0].id

    result_col, doc_col = st.columns([1, 1])

    with result_col:
        show_results(top_chunks)

    with doc_col:
        show_selected_chunk(selected_chunk, assets, documents, chunks)


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
