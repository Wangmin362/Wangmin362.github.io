


.PHONY: clean
clean:
	rm -rf public

# 初始化项目，下载子模块
.PHONY: init
init:
	git submodule update --init --recursive

.PHONY: run
run: clean
	hugo server --disableFastRender  --bind "0.0.0.0" -p 8020 -D
