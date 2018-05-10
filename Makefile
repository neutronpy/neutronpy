install:
	python setup.py install &> /dev/null

install-two:
	python2 setup.py install

install-three:
	python3 setup.py install

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo|plot_test.pdf|.coverage|test.out|.DS_Store$$)" | xargs rm -rf

test:
	pytest -p no:warnings --tb=short
	make clean

test-data:
	pytest -p no:warnings -v -x tests/test_data.py
	make clean

test-energy:
	pytest -p no:warnings -v -x tests/test_energy.py
	make clean

test-fitting:
	pytest -p no:warnings -v -x tests/test_fitting.py
	make clean

test-functions:
	pytest -p no:warnings -v -x tests/test_functions.py
	make clean

test-io:
	pytest -p no:warnings -v -x tests/test_io.py
	make clean

test-lattice:
	pytest -p no:warnings -v -x tests/test_lattice.py
	make clean

test-models:
	pytest -p no:warnings -v -x tests/test_models.py
	make clean

test-tof:
	pytest -p no:warnings -v -x tests/test_resolution_tof.py
	make clean

test-tas:
	pytest -p no:warnings -v -x tests/test_resolution_tas.py
	make clean

test-resolution:
	pytest -p no:warnings -v -x tests/test_resolution_tas.py tests/test_resolution_tof
	make clean

test-scattering:
	pytest -p no:warnings -v -x tests/test_scattering.py
	make clean

test-spurion:
	pytest -p no:warnings -v -x tests/test_spurion.py
	make clean

test-structure-factors:
	pytest -p no:warnings -v -x tests/test_structure_factors.py
	make clean

test-scans:
	pytest -p no:warnings -v -x tests/test_scans.py
	make clean

test-symmetry:
	pytest -p no:warnings -v -x tests/test_symmetry.py
	make clean

test-coverage:
	pytest --cov=neutronpy
	make clean

test-all:
	pytest -v
	make clean

pypi:
	find . | grep -E "(dist/|build/)" | xargs rm -rf
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*
	find . | grep -E "(dist/|build/)" | xargs rm -rf

test-docs:
	@$(MAKE) $(MAKE_FLAGS) -C doc html

docs:
	@$(MAKE) $(MAKE_FLAGS) -C doc io

clean-docs:
	find . | grep -E "(generated|_build$$)" | xargs rm -rf
