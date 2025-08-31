"""
WebAgents CRM & Analytics Skill

This skill provides agents with CRM and analytics capabilities, allowing them to:
- Track and manage contacts
- Record analytics events
- Query and analyze data
- Build user profiles and segments
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from uuid import uuid4

from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool


@dataclass
class Contact:
    """CRM Contact representation"""
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    lead_status: Optional[str] = 'new'  # new, contacted, qualified, converted, lost
    lead_stage: Optional[str] = None  # awareness, interest, consideration, intent, evaluation, purchase
    lead_score: Optional[int] = 0
    tags: Optional[List[str]] = None
    custom_attributes: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class AnalyticsEvent:
    """Analytics Event representation"""
    event_name: str
    event_category: Optional[str] = None
    event_action: Optional[str] = None
    event_label: Optional[str] = None
    event_value: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    email: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self):
        data = asdict(self)
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


class CRMAnalyticsSkill(Skill):
    """
    CRM & Analytics skill for WebAgents agents
    
    Provides comprehensive CRM and analytics capabilities for agents to track
    users, events, and build intelligent marketing campaigns.
    """
    
    def __init__(self, config: Dict[str, Any] = None, scope: str = "all"):
        super().__init__(config, scope)
        self.api_base_url = config.get('api_base_url', os.getenv('ROBUTLER_API_URL', 'https://webagents.ai/api'))
        self.api_key = config.get('api_key', os.getenv('ROBUTLER_API_KEY'))
        self.subject_type = config.get('subject_type', 'agent')
        self.subject_id = config.get('subject_id', str(uuid4()))
        self.namespace = config.get('namespace')
        self.session = None
        self.current_session_id = str(uuid4())
        
    async def initialize(self, agent):
        """Initialize the CRM skill"""
        from webagents.utils.logging import get_logger, log_skill_event
        
        self.agent = agent
        self.logger = get_logger('skill.webagents.crm', self.agent.name)
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
        )
        
        log_skill_event(self.agent.name, 'crm', 'initialized', {
            'has_api_key': bool(self.api_key),
            'api_base_url': self.api_base_url
        })
    
    # ============= CRM Contact Management =============
    
    @tool(description="Create or update a CRM contact", scope="all")
    async def create_or_update_contact(
        self,
        email: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        phone: Optional[str] = None,
        company: Optional[str] = None,
        job_title: Optional[str] = None,
        lead_status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_attributes: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Create or update a CRM contact
        
        Args:
            email: Contact email address
            first_name: First name
            last_name: Last name
            phone: Phone number
            company: Company name
            job_title: Job title
            lead_status: Lead status (new, contacted, qualified, converted, lost)
            tags: List of tags
            custom_attributes: Custom attributes dictionary
            notes: Notes about the contact
            
        Returns:
            Contact information
        """
        try:
            contact = Contact(
                email=email,
                first_name=first_name,
                last_name=last_name,
                phone=phone,
                company=company,
                job_title=job_title,
                lead_status=lead_status,
                tags=tags,
                custom_attributes=custom_attributes,
                notes=notes
            )
            
            data = {
                'subjectType': self.subject_type,
                'subjectId': self.subject_id,
                'namespace': self.namespace,
                **contact.to_dict()
            }
            
            async with self.session.post(f'{self.api_base_url}/crm/contacts', json=data) as resp:
                result = await resp.json()
                if resp.status >= 400:
                    raise Exception(f"Failed to create/update contact: {result.get('error')}")
                return {'success': True, 'contact': result}
                
        except Exception as e:
            self.logger.error(f"Failed to create/update contact: {e}")
            return {'success': False, 'error': str(e)}
    
    @tool(description="Get a contact by email", scope="all")
    async def get_contact(self, email: str, context=None) -> Dict[str, Any]:
        """
        Get a contact by email
        
        Args:
            email: Contact email address
            
        Returns:
            Contact information or None if not found
        """
        try:
            params = {
                'subjectType': self.subject_type,
                'subjectId': self.subject_id,
                'namespace': self.namespace,
                'email': email,
            }
            
            async with self.session.get(f'{self.api_base_url}/crm/contacts', params=params) as resp:
                result = await resp.json()
                if resp.status >= 400:
                    raise Exception(f"Failed to get contact: {result.get('error')}")
                
                contacts = result.get('contacts', [])
                contact = contacts[0] if contacts else None
                return {'success': True, 'contact': contact}
                
        except Exception as e:
            self.logger.error(f"Failed to get contact: {e}")
            return {'success': False, 'error': str(e)}
    
    @tool(description="Search contacts with filters", scope="all")
    async def search_contacts(
        self,
        lead_status: Optional[str] = None,
        lifecycle_stage: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = 'createdAt',
        sort_order: str = 'desc',
        context=None
    ) -> Dict[str, Any]:
        """
        Search contacts with filters
        
        Args:
            lead_status: Filter by lead status
            lifecycle_stage: Filter by lifecycle stage (waitlist, free, paying, deleted)
            limit: Maximum number of results
            offset: Offset for pagination
            sort_by: Sort field (createdAt, lastSeenAt, leadScore, email)
            sort_order: Sort order (asc, desc)
            
        Returns:
            Search results with contacts and pagination info
        """
        try:
            params = {
                'subjectType': self.subject_type,
                'subjectId': self.subject_id,
                'namespace': self.namespace,
                'limit': limit,
                'offset': offset,
                'sortBy': sort_by,
                'sortOrder': sort_order,
            }
            
            if lead_status:
                params['leadStatus'] = lead_status
            if lifecycle_stage:
                params['lifecycleStage'] = lifecycle_stage
            
            async with self.session.get(f'{self.api_base_url}/crm/contacts', params=params) as resp:
                result = await resp.json()
                if resp.status >= 400:
                    raise Exception(f"Failed to search contacts: {result.get('error')}")
                return {'success': True, **result}
                
        except Exception as e:
            self.logger.error(f"Failed to search contacts: {e}")
            return {'success': False, 'error': str(e)}
    
    # ============= Analytics Event Tracking =============
    
    @tool(description="Track an analytics event", scope="all")
    async def track_event(
        self,
        event_name: str,
        event_category: Optional[str] = None,
        event_action: Optional[str] = None,
        event_label: Optional[str] = None,
        event_value: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Track an analytics event
        
        Args:
            event_name: Name of the event
            event_category: Event category
            event_action: Event action
            event_label: Event label
            event_value: Numeric value associated with event
            properties: Event properties
            user_id: User ID
            email: User email
            
        Returns:
            Event tracking result
        """
        try:
            event = AnalyticsEvent(
                event_name=event_name,
                event_category=event_category,
                event_action=event_action,
                event_label=event_label,
                event_value=event_value,
                properties=properties,
                user_id=user_id,
                email=email,
                session_id=self.current_session_id
            )
            
            data = {
                'subjectType': self.subject_type,
                'subjectId': self.subject_id,
                'namespace': self.namespace,
                'source': 'api',
                **event.to_dict()
            }
            
            async with self.session.post(f'{self.api_base_url}/crm/events', json=data) as resp:
                result = await resp.json()
                if resp.status >= 400:
                    raise Exception(f"Failed to track event: {result.get('error')}")
                return {'success': True, 'event': result}
                
        except Exception as e:
            self.logger.error(f"Failed to track event: {e}")
            return {'success': False, 'error': str(e)}
    
    @tool(description="Get analytics events with filters", scope="all")
    async def get_events(
        self,
        event_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        context=None
    ) -> Dict[str, Any]:
        """
        Query analytics events
        
        Args:
            event_name: Filter by event name
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            limit: Maximum number of results
            
        Returns:
            Events and pagination info
        """
        try:
            params = {
                'subjectType': self.subject_type,
                'subjectId': self.subject_id,
                'namespace': self.namespace,
                'limit': limit,
            }
            
            if event_name:
                params['eventName'] = event_name
            if start_date:
                params['startDate'] = start_date
            if end_date:
                params['endDate'] = end_date
            
            async with self.session.get(f'{self.api_base_url}/crm/events', params=params) as resp:
                result = await resp.json()
                if resp.status >= 400:
                    raise Exception(f"Failed to get events: {result.get('error')}")
                return {'success': True, **result}
                
        except Exception as e:
            self.logger.error(f"Failed to get events: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
    
    def get_dependencies(self) -> List[str]:
        """Get skill dependencies"""
        return ['aiohttp']  # Required for HTTP client
