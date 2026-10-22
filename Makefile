UV ?= uv
IMAGE ?= fanpage-chatbot:local

.PHONY: install lint test widget check compose-config image container-smoke

install:
	$(UV) sync --frozen
	cd web/widget && npm ci

lint:
	$(UV) run ruff check .

test:
	$(UV) run pytest -q

widget:
	cd web/widget && npm test && npm run build
	git diff --exit-code -- app/web/static/widget.js

compose-config:
	POSTGRES_PASSWORD=compose-check docker compose config -q

check: lint test widget compose-config

image:
	docker build --target runtime --tag $(IMAGE) .

container-smoke: image
	./scripts/smoke_container.sh $(IMAGE)
	./scripts/smoke_compose.sh $(IMAGE)
