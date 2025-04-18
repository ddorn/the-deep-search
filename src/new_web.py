from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
from collections import defaultdict
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from constants import SYNC_PATTERN
from core_types import Asset, AssetType
from search import DocSearchResult, SearchEngine
from storage import setup_db

load_dotenv()

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


class UI:
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.debug = False

    def main(self):

        self.debug = st.sidebar.checkbox("Dev debug", value=False)

        stats = search_engine.stats()
        st.sidebar.write(
            f"Searching through **{stats.num_embeddings}** chunks for **{stats.num_documents}** documents."
        )

        with self.center_container(key="query"):
            query = st.text_input(
                "Query", value="delete", on_change=self.select_chunk, args=(None,),
                placeholder=self.search_engine.db.config.search_placeholder,
                label_visibility="collapsed",
            )

        with st.sidebar:
            nb_results = st.number_input("Max results", 1, 100, 10)
            threshold = st.number_input("Threshold", 0.0, 1.0, 0.2)

        if query:
            with st.spinner("Searching..."):
                start = time.time()
                results = self.search_engine.search(query, nb_results, threshold)
                end = time.time()

            nb_results = sum(len(result.chunks) for result in results)
            plural = "s" if nb_results != 1 else ""
            st.write(f":grey[*Found {nb_results} result{plural} in {end - start:.2f} seconds.*]")
            if nb_results == 0:
                st.info("No results found. You can try to lower the threshold in the left sidebar, or try a different query.")

            if len(results) == 0:
                self.show_splash_message()
                st.stop()

            if self.selected_chunk is None:
                self.selected_chunk = results[0].chunks[0].chunk.id

            results_col, doc_col = st.columns([1, 1])
            with results_col:
                self.show_results(results)

            with doc_col:
                self.show_selected_chunk()

        else:
            self.show_splash_message()

    def show_results(self, results: list[DocSearchResult]):
        for doc_result in results:
            st.markdown(f"### {doc_result.document.title}")
            for chunk_result in doc_result.chunks:
                content_col, button_col = st.columns([4, 1])
                with content_col:
                    text = ""
                    if chunk_result.path:
                        text += f"**{' ⟩ '.join(chunk_result.path)}** *{chunk_result.score:.3f}*\n\n"
                    else:
                        text += f"*{chunk_result.score:.3f}*\n\n"

                    text += "> " + chunk_result.nice_extract
                    st.markdown(text)
                    # st.markdown(chunk_result.chunk.content)
                with button_col:
                    st.button(
                        "👉",
                        key=f"select_{chunk_result.chunk.id}",
                        on_click=self.select_chunk,
                        args=(chunk_result.chunk.id,),
                    )

    def show_selected_chunk(self):
        if self.selected_chunk is None:
            st.info("Nothing to show, search and select a chunk first.")
            return

        chunk = self.search_engine.db.get_chunks([self.selected_chunk])[0]
        document = self.search_engine.db.get_document(chunk.document_id)
        assets = self.search_engine.db.get_assets_for_document(document.id)
        mark = re.search(SYNC_PATTERN, chunk.content).group("id")

        asset_by_type = defaultdict(list)
        for asset in assets:
            asset_by_type[asset.type].append(asset)


        if self.debug:
            st.write(f"### {document.title}")
            asset_type = st.radio(
                "Asset type to show",
                [type_ for type_ in asset_by_type],
                horizontal=True,
            )
            for asset in asset_by_type[asset_type]:
                self.show_asset(asset, mark)

        else:
            if document.url is not None:
                st.link_button(
                    label="Open",
                    url=document.url,
                    type="secondary",
                    icon=":material/open_in_new:",
                )

            audio_container = st.empty()

            if assets := asset_by_type.get(AssetType.NICE_MARKDOWN):
                mark_clicked = mark_linker(markdown=assets[0].path.read_text(), highlighted_mark=mark)
                try:
                    start_time = float(mark_clicked)
                except (ValueError, TypeError):
                    start_time = 0

            if AssetType.AUDIO_TO_DL in asset_by_type:
                audio_container.audio(asset_by_type[AssetType.AUDIO_TO_DL][0].content, start_time=start_time)

    def show_asset(self, asset: Asset, mark: str):
        if asset.type == AssetType.AUDIO_TO_DL:
            st.write(f"Audio: {asset.content}")
            st.audio(asset.content)
        elif asset.type in [AssetType.NICE_MARKDOWN, AssetType.SYNCED_TEXT_FILE]:
            mark_linker(markdown=asset.path.read_text(), highlighted_mark=mark)
            st.code(asset.path.read_text())
        elif asset.type == AssetType.STRUCTURE:
            st.json(json.loads(asset.path.read_text()))
        elif asset.type == AssetType.AUDIO_FILE:
            st.write(f"Audio at {asset.path}")
        elif asset.path is not None:
            st.code(asset.path.read_text())
        else:
            st.code(asset.content)

    def show_splash_message(self):
        extra_style = """div:has(.st-key-centered) { height: 100% !important; }"""
        with self.center_container(extra_style=extra_style):
            st.write(self.search_engine.db.config.splash_text)

    def center_container(self, key: str="", extra_style: str=""):
        style = """
<style>
    .st-key-centered{key} {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .st-key-centered{key} > div {
        max-width: 500px !important;
        width: 100%;
    }
    /* The first child of the centered container is the style, so we need to hide it */
    .st-key-centered{key} > div:first-child {
        display: none;
    }
    {extra_style}
</style>
""".replace("{key}", key).replace("{extra_style}", extra_style)

        container = st.container(key=f"centered{key}")
        container.write(style, unsafe_allow_html=True)
        return container

    def select_chunk(self, chunk_id: int):
        self.selected_chunk = chunk_id

    @property
    def selected_chunk(self):
        try:
            return st.session_state["selected_chunk"]
        except KeyError:
            return None

    @selected_chunk.setter
    def selected_chunk(self, chunk_id: int):
        st.session_state["selected_chunk"] = chunk_id


if __name__ == "__main__":
    env_config_path = os.getenv("DS_CONFIG")
    if env_config_path:
        db = setup_db(Path(env_config_path))
    else:
        db = setup_db()

    @st.cache_resource  # So it is shared across sessions and reruns.
    def get_global_cache():
        return {}

    search_engine = SearchEngine(db, get_global_cache())

    ui = UI(search_engine)
    ui.main()
