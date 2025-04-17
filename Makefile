.PHONY: web

web:
	uv run streamlit run src/new_web.py --server.runOnSave true
