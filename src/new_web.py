import json
import re
from collections import defaultdict

import streamlit as st
import streamlit.components.v1 as components

from constants import SYNC_PATTERN
from core_types import Asset, AssetType
from search import DocSearchResult, SearchEngine
from storage import setup_db

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
        stats = search_engine.stats()
        st.write(
            f"Searching through **{stats.num_embeddings}** chunks for **{stats.num_documents}** documents."
        )

        q_col, nb_results_col = st.columns([3, 1])
        with q_col:
            query = st.text_input(
                "Query", value="delete", on_change=self.select_chunk, args=(None,)
            )
        with nb_results_col:
            nb_results = st.number_input("Number of results", 1, 100, 5)
            self.debug = st.checkbox("Dev debug", value=False)

        if query:
            results = self.search_engine.search(query, nb_results)

            if self.selected_chunk is None:
                self.selected_chunk = results[0].chunks[0].chunk.id

            results_col, doc_col = st.columns([1, 1])
            with results_col:
                self.show_results(results)

            with doc_col:
                self.show_selected_chunk()

    def show_results(self, results: list[DocSearchResult]):
        for doc_result in results:
            st.markdown(f"### {doc_result.document.title}")
            for chunk_result in doc_result.chunks:
                content_col, button_col = st.columns([4, 1])
                with content_col:
                    st.markdown(f"**{chunk_result.score:.3f}** {chunk_result.nice_extract}")
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

        st.write(f"### {document.title}")

        if self.debug:
            asset_type = st.radio(
                "Asset type to show", [type_ for type_ in asset_by_type], horizontal=True
            )
            for asset in asset_by_type[asset_type]:
                self.show_asset(asset)

        else:
            if document.url is not None:
                st.link_button(
                    label="Open",
                    url=document.url,
                    type="secondary",
                    icon=":material/open_in_new:",
                )

            if AssetType.AUDIO_TO_DL in asset_by_type:
                st.audio(asset_by_type[AssetType.AUDIO_TO_DL][0].content)

            if assets := asset_by_type.get(AssetType.NICE_MARKDOWN):
                mark_linker(markdown=assets[0].path.read_text(), highlighted_mark=mark)

    def show_asset(self, asset: Asset):
        if asset.type == AssetType.AUDIO_TO_DL:
            st.write(f"Audio: {asset.content}")
            st.audio(asset.content)
        elif asset.type == AssetType.NICE_MARKDOWN:
            # mark_linker(markdown=asset.path.read_text())
            st.markdown(asset.path.read_text())
        elif asset.type == AssetType.STRUCTURE:
            st.json(json.loads(asset.path.read_text()))
        elif asset.type == AssetType.AUDIO_FILE:
            st.write(f"Audio at {asset.path}")
        elif asset.path is not None:
            st.code(asset.path.read_text())
        else:
            st.code(asset.content)

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

    db = setup_db()

    @st.cache_resource  # So it is shared across sessions and reruns.
    def get_search_engine():
        return SearchEngine(db)

    search_engine = get_search_engine()
    search_engine.db = db

    ui = UI(search_engine)
    ui.main()
