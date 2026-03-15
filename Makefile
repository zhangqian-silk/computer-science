.PHONY: default build dev preview view

default: dev

build:
	npm run docs:build

dev:
	npm run docs:dev

preview: build
	npm run docs:preview

view: dev
