.PHONY: default build dev preview view check

default: dev

build:
	npm run docs:build

dev:
	npm run docs:dev

preview: build
	npm run docs:preview

view: dev

# 设计系统一致性 + 主题对比度校验
check:
	npm run check
