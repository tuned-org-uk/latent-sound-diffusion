"""Pytest configuration for the ALD-SC test suite.

Shared helpers (factories, ``perturb_adaln``, constants) live in
``_helpers.py`` so they can be imported explicitly by test modules
(``from _helpers import ...``); ``tests`` is on ``pythonpath`` via
``pyproject.toml``.  This file is kept for future pytest fixture
definitions that need auto-discovery.
"""
