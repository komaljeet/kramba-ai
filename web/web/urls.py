from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def health(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health/", health),
    path("api/", include("api.urls")),
    # Minimal single-page chat UI (Issue #56), served at the site root.
    path("", include("chat.urls")),
]
