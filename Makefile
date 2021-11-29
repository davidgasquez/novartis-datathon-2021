.DEFAULT_GOAL := run

IMAGE_NAME := novartis-dathathon-2021:latest

build:
	docker build -t $(IMAGE_NAME) .

dev: build
	docker run --rm -it -p 8888:8888 -v $(PWD):/workspaces/novartis-datathon-2021 $(IMAGE_NAME) /bin/bash

notebook: build
	docker run --rm -it -p 8888:8888 -v $(PWD):/workspaces/novartis-datathon-2021 $(IMAGE_NAME)
