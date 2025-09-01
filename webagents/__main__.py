#!/usr/bin/env python3
"""
WebAgents CLI Entry Point

Command-line interface for the WebAgents framework.
"""

import sys
import argparse
from webagents.server.core.app import WebAgentsServer
from webagents.utils.logging import setup_logging


def main():
    """Main entry point for the webagents CLI."""
    parser = argparse.ArgumentParser(description="WebAgents - AI Agent Framework")
    parser.add_argument(
        "--server", 
        action="store_true", 
        help="Start the WebAgents server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    if args.server:
        # Start the server
        server = WebAgentsServer()
        server.run(host=args.host, port=args.port)
    else:
        print("WebAgents - AI Agent Framework")
        print("Use --server to start the server")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
