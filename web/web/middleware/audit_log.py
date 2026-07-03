"""
Placeholder audit-log middleware (Phase 0 scaffolding).

Phase 5 will extend this to persist an AuditLog row (user, path, method,
status, timestamp) for every generation-related request. Left as a no-op
passthrough for now so the middleware stack is wired up from the start.
"""


class AuditLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)
