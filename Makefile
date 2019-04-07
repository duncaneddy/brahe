.PHONY: docs
docs:
	travis-sphinx --outdir docs/build build --source docs/source

.PHONY: test
test:
	pytest --cov=brahe --durations=0