"""Single source of truth for the package version.

Kept in a tiny standalone module so it can be read without importing the
package (and triggering its dependencies). `pyproject.toml` mirrors this
string and the top-level `VERSION` file does too. Bump all three on release.
"""

__version__ = "0.0.1"
