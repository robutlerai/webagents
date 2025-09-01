"""
CRM & Analytics Skill Package - WebAgents Platform Integration

CRM and analytics skill for WebAgents platform.
Provides contact management and event tracking capabilities.
"""

from .skill import (
    CRMAnalyticsSkill,
    Contact,
    AnalyticsEvent
)

__all__ = [
    "CRMAnalyticsSkill",
    "Contact", 
    "AnalyticsEvent"
]
