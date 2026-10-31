PY := server/.venv/bin/python
.PHONY: setup precompute test dev server game demo push-triologay
setup:
	cd server && uv venv -q --python 3.13 .venv && uv pip install -q -r pyproject.toml
	cd game && npm install
precompute:
	cd server && .venv/bin/python -m pipelines.run_all
test:
	cd server && .venv/bin/python -m pytest -q
server:
	cd server && .venv/bin/uvicorn app.main:app --port 8000
game:
	cd game && npm run dev
dev:
	$(MAKE) -j2 server game
push-triologay:
	rsync -az --exclude .venv --exclude var --exclude node_modules --exclude .git ./ root@triologay:/root/orbital-dodge/
