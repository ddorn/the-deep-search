.PHONY: web

web: 
	uv run streamlit run src/web.py --server.runOnSave true
