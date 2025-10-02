"""
WebAgents V2.0 Logging System

Comprehensive logging with color-coded agent names, structured output,
and context-aware logging for agents, skills, and server components.
"""

import logging
import sys
import time
from typing import Optional, Dict, Any
from colorama import Fore, Back, Style, init
from datetime import datetime
from contextvars import ContextVar

# Initialize colorama for cross-platform color support
init(autoreset=True)

# Context variables for logging
CURRENT_AGENT: ContextVar[Optional[str]] = ContextVar('current_agent', default=None)
CURRENT_REQUEST_ID: ContextVar[Optional[str]] = ContextVar('current_request_id', default=None)
CURRENT_USER_ID: ContextVar[Optional[str]] = ContextVar('current_user_id', default=None)

# Color mapping for different agent types/names
AGENT_COLORS = {
    # Core system colors
    'system': Fore.CYAN,
    'server': Fore.BLUE,
    'router': Fore.MAGENTA,
    
    # Agent name-based colors (hash-based for consistency)
    'default': Fore.GREEN,
}

def get_agent_color(agent_name: str) -> str:
    """Get consistent color for agent name based on hash"""
    if agent_name in AGENT_COLORS:
        return AGENT_COLORS[agent_name]
    
    # Generate consistent color based on agent name hash
    # Using a variety of colors for better visual distinction
    colors = [
        Fore.GREEN,
        Fore.CYAN,
        Fore.YELLOW,
        Fore.BLUE,
        Fore.MAGENTA,
        Fore.LIGHTGREEN_EX,
        Fore.LIGHTCYAN_EX,
        Fore.LIGHTYELLOW_EX,
        Fore.LIGHTBLUE_EX,
        Fore.LIGHTMAGENTA_EX,
    ]
    color_index = hash(agent_name) % len(colors)
    return colors[color_index]

class WebAgentsFormatter(logging.Formatter):
    """Custom formatter with color-coded agent names and structured output"""
    
    def format(self, record):
        # Get context information
        agent_name = CURRENT_AGENT.get() or getattr(record, 'agent_name', None)
        request_id = CURRENT_REQUEST_ID.get() or getattr(record, 'request_id', None)
        user_id = CURRENT_USER_ID.get() or getattr(record, 'user_id', None)
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Format level with color
        level_colors = {
            'DEBUG': Fore.WHITE,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE
        }
        level_color = level_colors.get(record.levelname, Fore.WHITE)
        colored_level = f"{level_color}{record.levelname:8s}{Style.RESET_ALL}"
        
        # Format agent name with color
        if agent_name:
            agent_color = get_agent_color(agent_name)
            colored_agent = f"{agent_color}{agent_name:15s}{Style.RESET_ALL}"
        else:
            # Try to extract agent name from logger name (e.g., openlicense-image.tool)
            logger_parts = record.name.split('.')
            if len(logger_parts) > 0 and '-' in logger_parts[0]:
                potential_agent = logger_parts[0]
                agent_color = get_agent_color(potential_agent)
                colored_agent = f"{agent_color}{potential_agent[:15]:15s}{Style.RESET_ALL}"
            else:
                colored_agent = f"{'system':15s}"
        
        # Format component (logger name)
        component = record.name.split('.')[-1] if '.' in record.name else record.name
        # Truncate very long component names
        if len(component) > 12:
            component = component[:12]
        
        # Build context string
        context_parts = []
        if request_id:
            context_parts.append(f"req={request_id[:8]}")
        if user_id:
            context_parts.append(f"user={user_id[:8]}")
        
        context_str = f" [{':'.join(context_parts)}]" if context_parts else ""
        
        # Format the log message
        message = record.getMessage()
        
        # Build final log line
        log_line = f"{Fore.WHITE}{timestamp}{Style.RESET_ALL} {colored_level} {colored_agent} {Fore.BLUE}{component:12s}{Style.RESET_ALL}{context_str} {message}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += '\n' + self.formatException(record.exc_info)
        
        return log_line

class AgentContextAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes agent context"""
    
    def __init__(self, logger, agent_name: str):
        super().__init__(logger, {})
        self.agent_name = agent_name
    
    def process(self, msg, kwargs):
        # Add agent context to log record
        extra = kwargs.get('extra', {})
        extra['agent_name'] = self.agent_name
        
        # Add current context if available
        request_id = CURRENT_REQUEST_ID.get()
        user_id = CURRENT_USER_ID.get()
        
        if request_id:
            extra['request_id'] = request_id
        if user_id:
            extra['user_id'] = user_id
            
        kwargs['extra'] = extra
        return msg, kwargs

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup the WebAgents logging system"""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger('webagents')
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with color formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(WebAgentsFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler if specified (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)-20s [%(agent_name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Disable propagation to avoid duplicate logs
    root_logger.propagate = False
    
    # Set levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Configure LiteLLM logging to use our formatter
    litellm_logger = logging.getLogger('LiteLLM')
    litellm_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    # Clear any existing handlers from LiteLLM
    litellm_logger.handlers.clear()
    # Add our console handler with formatter
    litellm_handler = logging.StreamHandler(sys.stdout)
    litellm_handler.setFormatter(WebAgentsFormatter())
    litellm_logger.addHandler(litellm_handler)
    litellm_logger.propagate = False
    
    # Also configure the root logger to catch any unconfigured loggers
    # This will catch any dynamically created loggers that don't have handlers
    root = logging.getLogger()
    if not root.handlers:
        # Only add handler if root doesn't have one already
        root_handler = logging.StreamHandler(sys.stdout)
        root_handler.setFormatter(WebAgentsFormatter())
        root_handler.setLevel(numeric_level)
        root.addHandler(root_handler)
    else:
        # Update existing root handlers to use our formatter
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(WebAgentsFormatter())
                handler.setLevel(numeric_level)

def get_logger(name: str, agent_name: Optional[str] = None) -> logging.Logger:
    """Get a logger for a specific component, optionally with agent context"""
    
    logger_name = f"webagents.{name}" if not name.startswith('webagents') else name
    logger = logging.getLogger(logger_name)
    
    if agent_name:
        return AgentContextAdapter(logger, agent_name)
    
    return logger

def set_agent_context(agent_name: str, request_id: Optional[str] = None, user_id: Optional[str] = None):
    """Set the current agent context for logging"""
    CURRENT_AGENT.set(agent_name)
    if request_id:
        CURRENT_REQUEST_ID.set(request_id)
    if user_id:
        CURRENT_USER_ID.set(user_id)

def clear_agent_context():
    """Clear the current agent context"""
    CURRENT_AGENT.set(None)
    CURRENT_REQUEST_ID.set(None)
    CURRENT_USER_ID.set(None)

# Convenience functions for different log types
def log_agent_action(agent_name: str, action: str, details: Optional[Dict[str, Any]] = None):
    """Log an agent action with structured data"""
    logger = get_logger('agent.action', agent_name)
    details_str = f" {details}" if details else ""
    logger.info(f"{action}{details_str}")

def log_skill_event(agent_name: str, skill_name: str, event: str, details: Optional[Dict[str, Any]] = None):
    """Log a skill event"""
    logger = get_logger(f'skill.{skill_name}', agent_name)
    details_str = f" {details}" if details else ""
    logger.info(f"{event}{details_str}")

def log_tool_execution(agent_name: str, tool_name: str, duration_ms: int, success: bool = True):
    """Log tool execution with timing"""
    logger = get_logger('tool.execution', agent_name)
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Tool {tool_name} {status} ({duration_ms}ms)")

def log_handoff(agent_name: str, handoff_type: str, target: str, success: bool = True):
    """Log handoff execution"""
    logger = get_logger('handoff', agent_name)
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Handoff {handoff_type} -> {target} {status}")

def log_server_request(endpoint: str, method: str, status_code: int, duration_ms: int, agent_name: Optional[str] = None):
    """Log server request with timing"""
    logger = get_logger('server.request', agent_name)
    logger.info(f"{method} {endpoint} {status_code} ({duration_ms}ms)")

# Performance logging utilities
class LogTimer:
    """Context manager for timing operations with automatic logging"""
    
    def __init__(self, operation_name: str, logger: logging.Logger, level: int = logging.INFO):
        self.operation_name = operation_name
        self.logger = logger
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.time() - self.start_time) * 1000)
        if exc_type:
            self.logger.error(f"{self.operation_name} FAILED ({duration_ms}ms): {exc_val}")
        else:
            self.logger.log(self.level, f"{self.operation_name} completed ({duration_ms}ms)")

def timer(operation_name: str, agent_name: Optional[str] = None, level: int = logging.INFO):
    """Create a timing context manager"""
    logger = get_logger('performance', agent_name)
    return LogTimer(operation_name, logger, level)

def configure_external_logger(logger_name: str, level: Optional[str] = None) -> None:
    """Configure an external logger to use our structured format
    
    Args:
        logger_name: Name of the logger to configure (e.g., 'litellm', 'openlicense')
        level: Optional log level to set (defaults to current root level)
    """
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Get the current root level if not specified
    if level is None:
        root_logger = logging.getLogger('webagents')
        numeric_level = root_logger.level
    else:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Add our formatted handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(WebAgentsFormatter())
    handler.setLevel(numeric_level)
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    logger.propagate = False

def capture_all_loggers() -> None:
    """Ensure all existing loggers use our structured format
    
    This function should be called after all modules are imported
    to ensure any dynamically created loggers are properly configured.
    """
    root_formatter = WebAgentsFormatter()
    root_level = logging.getLogger('webagents').level
    
    # Get all existing loggers
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        
        # Skip if it's already a webagents logger
        if name.startswith('webagents'):
            continue
        
        # Skip if it has handlers (already configured)
        if logger.handlers:
            # Update handlers to use our formatter
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(root_formatter)
        else:
            # Add our handler if no handlers exist
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(root_formatter)
            handler.setLevel(root_level)
            logger.addHandler(handler)
            logger.propagate = False

# Custom logger class that automatically uses our formatter
class WebAgentsLogger(logging.Logger):
    """Custom logger that automatically gets our formatter"""
    
    def __init__(self, name):
        super().__init__(name)
        # Automatically add our handler if this is a new logger
        if not self.handlers and not name.startswith('webagents'):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(WebAgentsFormatter())
            # Get the root level
            root_level = logging.getLogger('webagents').level if logging.getLogger('webagents').level else logging.INFO
            handler.setLevel(root_level)
            self.addHandler(handler)
            self.setLevel(root_level)
            self.propagate = False

# Set our custom logger class as the default
logging.setLoggerClass(WebAgentsLogger)

# Initialize logging on import (can be reconfigured later)
setup_logging() 