.PHONY: install

install: install-dependencies install-cec2017

install-dependencies:
	pip install -r requirements.txt

install-cec2017:
	git clone https://github.com/tilleyd/cec2017-py.git
	cd cec2017-py && pip install .