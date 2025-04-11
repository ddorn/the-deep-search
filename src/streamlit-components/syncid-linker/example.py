import streamlit as st
from syncid_linker import syncid_linker

value_input = st.text_area("Enter markdown:", value="")
v = syncid_linker(markdown=value_input, mapping={14: 1, 27: 2, 55: 3, 77: 4, 111: 5})
