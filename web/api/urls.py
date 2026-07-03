from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from django.urls import include, path

from .views import ConversationViewSet, LogoutView, ModelVersionViewSet, RegisterView

router = DefaultRouter()
router.register("conversations", ConversationViewSet, basename="conversation")
router.register("models", ModelVersionViewSet, basename="modelversion")

urlpatterns = [
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("", include(router.urls)),
]
