from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from django.urls import include, path

from .account import (
    AccountDeleteView,
    AccountExportView,
    TwoFactorConfirmView,
    TwoFactorSetupView,
    TwoFactorTokenObtainPairView,
)
from .views import (
    ConversationViewSet,
    LogoutView,
    ModelVersionViewSet,
    RegisterView,
    TrainingJobViewSet,
)

router = DefaultRouter()
router.register("conversations", ConversationViewSet, basename="conversation")
router.register("models", ModelVersionViewSet, basename="modelversion")
router.register("training", TrainingJobViewSet, basename="training")

urlpatterns = [
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", TwoFactorTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("auth/2fa/setup/", TwoFactorSetupView.as_view(), name="2fa_setup"),
    path("auth/2fa/confirm/", TwoFactorConfirmView.as_view(), name="2fa_confirm"),
    path("account/export/", AccountExportView.as_view(), name="account_export"),
    path("account/", AccountDeleteView.as_view(), name="account_delete"),
    path("", include(router.urls)),
]
