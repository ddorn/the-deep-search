import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import BaseModel
from pydantic import Field
from pydantic_core import PydanticUndefinedType
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from config import setup_config
from constants import SYNC_PATTERN
from core_types import Asset, AssetType
from search import DocSearchResult, SearchEngine
from storage import Database
from sources import BUILT_IN_SOURCES
import streamlit_react_jsonschema as srj

load_dotenv()

st.set_page_config(
    page_title="The Deep Search",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


debug = st.sidebar.checkbox("Debug", value=False)

@st.cache_resource
def get_sources():
    return []

sources = get_sources()


class PydanticForm[T: type[BaseModel]]:
    def __init__(self, model: type[T]):
        self.model = model
        self.json_schema = model.model_json_schema()
        self.defs = self.json_schema.get("$defs", {})

    def render_field(self, schema: dict, key: str, field_name: str):
        schema = self.unwrap_ref(schema)

        field_type = schema.get("type")
        field_format = schema.get("format")
        default = schema.get("default")

        kwargs = dict(
            value=default,
            key=key,
            label_visibility="collapsed",
            label=key,
        )

        if field_type == "string":
            if field_format == "date-time":
                return st.date_input(**kwargs)
            elif field_format == "multiline":
                return st.text_area(**kwargs)
            else:
                return st.text_input(**kwargs)
        elif field_type == "number":
            return st.number_input(**kwargs)
        elif field_type == "object":
            return self.render_object(schema, key=key)
        elif field_type == "null":
            st.write(f"{self.nice_title(schema, field_name)} is null.")
            return None
        elif "anyOf" in schema:
            return self.render_any_of(schema, key=key, field_name=field_name)
        else:
            st.error(f"Unsupported field type: {field_type}")
            return None

    def render_any_of(self, schema: dict, key: str, field_name: str):
        schema = self.unwrap_ref(schema)
        option_names = []
        for i, option in enumerate(schema["anyOf"]):
            option_names.append(self.nice_title(option, default=f"Option {i}"))

        selected = st.radio(label=key, options=option_names, index=0, label_visibility="collapsed", key=key + "-select", horizontal=True)
        selected_index = option_names.index(selected)
        selected_option = schema["anyOf"][selected_index]
        return self.render_field(selected_option, key=key + f"-{selected_index}", field_name=f"{field_name}-{selected}")

    def render_object(self, schema: dict, key: str):
        form_data = {}
        for field_name, field in schema.get("properties", {}).items():
            field = self.unwrap_ref(field)

            title = self.nice_title(field, field_name)
            description = field.get("description", "")

            cols = st.columns(1 + debug)
            if debug:
                with cols[1]:
                    st.write(field)

            with cols[0]:
                header = [f"**{title}**", description]
                st.write("  \n".join(header))
                form_data[field_name] = self.render_field(field, key=f"{key}_{field_name}", field_name=field_name)

        # If object has no properties, we prompt for any json data, as text
        if not form_data:
            json_data = st.text_area(
                label="JSON data",
                placeholder="{\n  \"key\": \"value\"\n}",
                key=key,
            )

            try:
                form_data = json.loads(json_data)
            except json.JSONDecodeError:
                return {}

        return form_data

    def render(self, key: str) -> T | None:
        """Render a form for a Pydantic model."""
        form_data = self.render_object(self.json_schema, key=key + "_root")

        submit_button = st.button(label="Submit", key=key + "_submit")

        if submit_button:
            # Validate the form data against the model
            try:
                validated_data = self.model.model_validate(form_data)
                st.success("Form submitted successfully!")
                return validated_data
            except Exception as e:
                st.error(f"Error: {e}")
                return None
        return None


    def unwrap_ref(self, schema: dict):
        """Get the reference from the schema."""
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            return self.defs[ref_name]
        return schema

    def nice_title(self, schema: dict, field_name: str | None = None, default: str = "Untitled"):
        """Get the title from the schema."""
        schema = self.unwrap_ref(schema)

        if "title" in schema:
            return schema["title"]

        if field_name is not None:
            return field_name.replace("_", " ").capitalize()

        if "format" in schema:
            return schema["format"].capitalize()
        if "type" in schema:
            return schema["type"].capitalize()
        return default



if debug:
    for source_name, source_class in BUILT_IN_SOURCES.items():
        config_class: type[BaseModel] = source_class.CONFIG_TYPE
        st.subheader(source_name)
        PydanticForm(config_class).render(source_name)

display_names = [source.DISPLAY_NAME for source in BUILT_IN_SOURCES.values()]
descriptions = [source.DESCRIPTION for source in BUILT_IN_SOURCES.values()]

st.title("New Source Wizard")
new_source = st.radio(
    "New source",
    options=display_names,
    captions=descriptions,
    index=None,
)
if new_source is None:
    st.stop()

st.header(f"⚙️ {new_source}")
source_class = next(
    source for source in BUILT_IN_SOURCES.values() if source.DISPLAY_NAME == new_source
)
source_config_class = source_class.CONFIG_TYPE
with st.container(border=True):
    config = PydanticForm(source_config_class).render(new_source)
if config:
    st.write("Config:")
    st.code(config.model_dump_json())
