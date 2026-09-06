"""API route registration.

Keep framework-specific route wiring in this module so the application core
remains independent of its HTTP transport.
"""


def register_routes(application: object) -> object:
    """Register API routes on *application* and return it.

    The concrete web framework is intentionally selected by the application's
    composition layer rather than by the agent domain package.
    """
    return application

