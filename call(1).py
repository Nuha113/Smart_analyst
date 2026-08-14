"""Stable messaging import path without notebook code or embedded credentials."""

from backend.app.services.messaging_service import MessagingService
from backend.app.services.template_service import TemplateService

__all__ = ["MessagingService", "TemplateService"]
