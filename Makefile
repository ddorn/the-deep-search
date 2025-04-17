UV ?= uv
FRONTEND_DIR = ./src/streamlit-components/mark_linker/mark_linker/frontend
STREAMLIT_ARGS = --browser.gatherUsageStats false

.PHONY: web build-frontend serve-frontend serve-backend

web:
	$(UV) run streamlit run src/new_web.py --server.runOnSave true $(STREAMLIT_ARGS)

build-frontend:
	@if command -v pnpm > /dev/null; then \
		cd $(FRONTEND_DIR); pnpm build; \
    else \
        nix develop $(FRONTEND_DIR) -c sh -c "cd $(FRONTEND_DIR); pnpm build"; \
    fi

serve-frontend:
	$(UV) run streamlit run src/new_web.py $(STREAMLIT_ARGS)

serve-backend:
	$(UV) run src/main.py  main
