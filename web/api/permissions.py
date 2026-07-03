from rest_framework.permissions import SAFE_METHODS, BasePermission


class IsAdminOrReadOnly(BasePermission):
    """Any authenticated user can list/retrieve (GET/HEAD/OPTIONS); only
    staff can create/update/delete. Used for the ModelVersion registry
    (Issue #30 Phase 6) - regular users need to see which model versions
    exist to pick one when creating a conversation, but only an admin
    registers a new checkpoint."""

    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True
        return bool(request.user and request.user.is_staff)


class IsConversationOwner(BasePermission):
    """Object-level check: a user may only touch their own conversations.

    Combine with IsAuthenticated at the view level - this only runs on
    detail actions (retrieve/update/delete), not list/create, which is why
    ConversationViewSet.get_queryset() *also* filters by owner: without that,
    a user could still discover other users' conversation IDs existed via
    list, even if denied on retrieve.
    """

    def has_object_permission(self, request, view, obj):
        return obj.owner_id == request.user.id
