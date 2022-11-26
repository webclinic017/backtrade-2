.PHONY: docs
docs:
	sphinx-apidoc -o docs/source src/backtrade -M
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	explorer docs\build\html\index.html
