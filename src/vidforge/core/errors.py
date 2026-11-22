"""Errors the user can understand. `code` picks a translated message in the GUI; the English
text stays for logs and the command line."""

from __future__ import annotations


class FriendlyError(ValueError):
    def __init__(self, code: str, message: str, **params):
        super().__init__(message)
        self.code = code
        self.params = params
