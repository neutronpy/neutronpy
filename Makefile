install:
	python2 setup.py install
	python3 setup.py install

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo|plot_test.pdf|test.out$$)" | xargs rm -rf
	
test:
	python3 setup.py test
	
test-all:
	python2 setup.py test
	python3 setup.py test
	
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