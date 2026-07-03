"""
Minimal hand-rolled Content-Security-Policy middleware (Issue #30's "CSP
headers for web client" requirement). This API mostly returns JSON, so the
policy only really matters for the Django admin and DRF's browsable API -
still worth setting since both render HTML and a permissive default CSP
would otherwise apply to them. Not using django-csp here to avoid pulling in
a whole extra dependency for one header.
"""

CSP_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",  # DRF's browsable API inlines some CSS
        "img-src 'self' data:",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)


class ContentSecurityPolicyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response.setdefault("Content-Security-Policy", CSP_POLICY)
        return response
