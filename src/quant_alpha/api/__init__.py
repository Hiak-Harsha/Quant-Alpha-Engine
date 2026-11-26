"""FastAPI backend serving Quant Alpha models, live scans, paper simulator, and React SPA."""

from quant_alpha.api.app import app, create_app

__all__ = ["app", "create_app"]
