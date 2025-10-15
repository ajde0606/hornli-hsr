.PHONY: develop install clean

develop:
	pip install -e . --config-settings editable_mode=compat --no-build-isolation

# Normal install
install:
	pip install .

# Remove build artifacts
clean:
	rm -rf build dist *.egg-info
