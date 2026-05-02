"""Top-level orchestration scripts for lerobot-bench.

These are CLI entrypoints rather than library code; they live outside
``src/`` because they are what the operator invokes directly. The
package marker exists so tests can import them as
``from scripts.<name> import ...`` without path manipulation.
"""
