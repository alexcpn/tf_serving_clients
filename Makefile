# Parameters
MKDIR=mkdir
SHELL:=/bin/bash
PYTHON=python

all: clean build 

build:
	$(MKDIR) -p ./cnn_client/cnncli_generated
	touch ./cnn_client/cnncli_generated/__init__.py
	$(PYTHON) -m grpc_tools.protoc  -I ./cnn_client --python_out=./cnn_client/cnncli_generated  --grpc_python_out=./cnn_client/cnncli_generated string_int_label_map.proto
	
	rm -rf dist
	mkdir dist
	$(PYTHON) setup.py sdist bdist_wheel
	
clean:
	rm -rf build dist tfservingclinet.egg-info ./cnn_client/cnncli_generated
