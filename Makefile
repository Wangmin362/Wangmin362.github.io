
.PHONY: clean
clean:
	rm -rf public

.PHONY: run
run:
	hugo server --disableFastRender  --bind "0.0.0.0" -p 8020 -D -v --debug
