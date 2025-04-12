import streamlit as st
from mark_linker import mark_linker

value_input = st.text_area("Enter markdown:", value="")
v = mark_linker(markdown=value_input, highlighted_mark="10")
