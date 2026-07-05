"""
Serves the minimal single-page chat UI (Issue #56).

Just renders a static template; all data flows client-side through the JSON API
(JWT in the Authorization header) and the WebSocket streaming endpoint, so this
view needs no server-side auth or context — the page's own JS handles login and
redirects. Kept as a plain Django view (not DRF) because it returns HTML, not
JSON, and must be reachable before the user has a token.
"""
from django.views.generic import TemplateView


class ChatUIView(TemplateView):
    template_name = "chat/index.html"
