# Normal commands for any laptop.  Everything routes through bootstrap.py,
# which repairs the environment before it runs anything - so these work on a
# clean clone, and none of them needs to know where the virtualenv lives.
#
#   make setup     one-time: environment, dependencies, model, verify
#   make test      engine self-test (no camera needed)
#   make run       live trainer on the webcam
#   make doctor    diagnose the environment, change nothing
#   make demo      scripted practice session, then the progress report
#   make dataset   fetch the dataset, refit the reference, revalidate it
#   make docker    build the container and run the verification suite
#   make clean     remove generated files (keeps the dataset)

PY ?= python

.PHONY: help setup doctor test run calibrate demo dataset docker docker-session clean

help:
	@echo "make setup | test | run | calibrate | doctor | demo | dataset | docker | clean"

setup:
	$(PY) bootstrap.py

doctor:
	$(PY) bootstrap.py --check

test:
	$(PY) bootstrap.py --test

run:
	$(PY) bootstrap.py --run

calibrate:
	$(PY) bootstrap.py --calibrate

demo:
	$(PY) bootstrap.py --demo
	$(PY) bootstrap.py --report
	$(PY) bootstrap.py --frames

dataset:
	$(PY) bootstrap.py --dataset
	$(PY) bootstrap.py --fit
	$(PY) bootstrap.py --validate

docker:
	docker compose run --rm verify

docker-session:
	docker compose run --rm session

clean:
	rm -rf __pycache__ */__pycache__ data/demo data/snapshots data/sessions.db
	rm -f data/samples/*.mp4 data/samples/shot_*.png
