VENV = .venv
PYTEST = $(VENV)/bin/pytest
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt;

run: $(VENV)/bin/activate
	$(PYTHON) test.py;

test: $(VENV)/bin/activate
	$(PYTEST) -vv;

clean:
	rm -rf __pycache__
	rm -rf venv;