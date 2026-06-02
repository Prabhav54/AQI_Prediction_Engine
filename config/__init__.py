# Marker file — makes `config/` a proper Python package.
# The original repo accidentally named this file `__init___.py` (three
# trailing underscores) which left `config` as an implicit-namespace
# package and broke editable installs on some setups.