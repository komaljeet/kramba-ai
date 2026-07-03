from django.contrib import admin

from .models import AuditLog, Conversation, Message, ModelVersion


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("tag", "is_active", "created_by", "created_at")
    list_filter = ("is_active",)


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "title", "model_version", "is_archived", "updated_at")
    list_filter = ("is_archived", "model_version")
    search_fields = ("title", "owner__username")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "created_at")
    list_filter = ("role",)


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("action", "user", "path", "method", "status_code", "created_at")
    list_filter = ("action", "method", "status_code")
    readonly_fields = [f.name for f in AuditLog._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
