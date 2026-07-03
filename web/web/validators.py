"""
Django AUTH_PASSWORD_VALIDATORS entries beyond the stock ones - see
web/settings/base.py.
"""
import hashlib
import logging
import urllib.error
import urllib.request

from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class BreachedPasswordValidator:
    """Checks the password against HaveIBeenPwned's Pwned Passwords range
    API using k-anonymity - only a 5-character SHA-1 prefix ever leaves this
    server, never the password or its full hash (see
    https://haveibeenpwned.com/API/v3#PwnedPasswords).

    Fails open (allows the password, logs a warning) on any network error -
    a third-party API outage must not be able to block registration or
    password changes entirely.
    """

    API_URL = "https://api.pwnedpasswords.com/range/{prefix}"
    TIMEOUT_SECONDS = 3

    def validate(self, password, user=None):
        sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
        prefix, suffix = sha1[:5], sha1[5:]

        try:
            with urllib.request.urlopen(
                self.API_URL.format(prefix=prefix), timeout=self.TIMEOUT_SECONDS
            ) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, OSError, ValueError):
            logger.warning("Password breach check unavailable (network error) - allowing password")
            return

        for line in body.splitlines():
            candidate_suffix, _, _count = line.partition(":")
            if candidate_suffix == suffix:
                raise ValidationError(
                    "This password has appeared in a known data breach. Please choose a different password.",
                    code="password_breached",
                )

    def get_help_text(self):
        return "Your password must not have appeared in a known data breach."
