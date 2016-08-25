install:
	python2 setup.py install
	python3 setup.py install

install-two:
	python2 setup.py install

install-three:
	python3 setup.py install

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo|plot_test.pdf|.coverage|test.out|.DS_Store$$)" | xargs rm -rf
	
test:
	pytest --tb=short

test-data:
	pytest -v -x tests/test_data.py

test-energy:
	pytest -v -x tests/test_energy.py

test-fitting:
	pytest -v -x tests/test_fitting.py

test-functions:
	pytest -v -x tests/test_functions.py

test-io:
	pytest -v -x tests/test_io.py

test-lattice:
	pytest -v -x tests/test_lattice.py

test-models:
	pytest -v -x tests/test_models.py

test-resolution:
	pytest -v -x tests/test_resolution.py

test-scattering:
	pytest -v -x tests/test_scattering.py

test-spurion:
	pytest -v -x tests/test_spurion.py

test-structure-factors:
	pytest -v -x tests/test_structure_factors.py

test-symmetry:
	pytest -v -x tests/test_symmetry.py

test-coverage:
	pytest --cov=neutronpy

test-all:
	pytest -v

pypi:
	python3 setup.py sdist upload -r pypi
	python3 setup.py bdist_wheel upload -r pypi
	find . | grep -E "(plot_test.pdf|test.out$$)" | xargs rm -rf

test-docs:
	@$(MAKE) $(MAKE_FLAGS) -C doc html

docs:
	@$(MAKE) $(MAKE_FLAGS) -C doc io

clean-docs:
	find . | grep -E "(generated|_build$$)" | xargs rm -rf
