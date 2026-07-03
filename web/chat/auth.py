"""
JWT auth for WebSocket connections. Channels has no concept of the
Authorization header dance HTTP uses, so the access token is passed as a
query parameter instead: `wss://.../ws/chat/<id>/?token=<access_token>`.

Rejects the connection (AnonymousUser, scope["user"] unauthenticated) rather
than raising, so consumers can uniformly check `scope["user"].is_authenticated`
and close(4401) - see Phase 4 / test_websocket.py.
"""
from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

from django.contrib.auth.models import AnonymousUser

_jwt_auth = JWTAuthentication()


@database_sync_to_async
def _get_user_from_token(raw_token: str):
    try:
        validated_token = _jwt_auth.get_validated_token(raw_token)
        return _jwt_auth.get_user(validated_token)
    except (InvalidToken, TokenError):
        return AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        query_string = parse_qs(scope.get("query_string", b"").decode())
        token = query_string.get("token", [None])[0]
        scope["user"] = await _get_user_from_token(token) if token else AnonymousUser()
        return await super().__call__(scope, receive, send)
