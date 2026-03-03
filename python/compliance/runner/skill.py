"""
TestRunnerSkill - Agentic compliance test runner.

This skill provides tools for an AI agent to run compliance tests
against WebAgents SDK implementations.
"""

import json
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from webagents.agents.agent import BaseAgent
from webagents.agents.skills.skill import Skill, tool


@dataclass
class TestResult:
    """Result of a single test case."""
    name: str
    passed: bool
    assertions: List[Dict[str, Any]]
    details: str
    duration_ms: float


@dataclass
class TestSuiteResult:
    """Result of a full test suite."""
    name: str
    test_file: str
    cases: List[TestResult]
    passed: int
    failed: int
    skipped: int


class TestRunnerSkill(Skill):
    """
    Skill that provides tools for running compliance tests.
    
    The test runner agent uses these tools to:
    1. Load test specifications from markdown files
    2. Execute HTTP requests against the SDK under test
    3. Validate responses against natural language and strict assertions
    4. Report detailed test results
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        cache_dir: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestSuiteResult] = []
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def shutdown(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @tool(description="Send HTTP request to the SDK server under test")
    async def http_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the SDK server.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: URL path (e.g., /chat/completions)
            body: Request body (for POST/PUT)
            headers: Additional headers
            stream: Whether to expect SSE streaming response
            
        Returns:
            Response dict with status, headers, and body
        """
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        
        try:
            async with session.request(
                method=method.upper(),
                url=url,
                json=body if body else None,
                headers=request_headers,
            ) as response:
                if stream and response.content_type == "text/event-stream":
                    # Collect SSE chunks
                    chunks = []
                    async for line in response.content:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: "):
                            data = decoded[6:]
                            if data != "[DONE]":
                                try:
                                    chunks.append(json.loads(data))
                                except json.JSONDecodeError:
                                    chunks.append({"raw": data})
                    
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "body": chunks,
                        "format": "sse",
                    }
                else:
                    try:
                        body_data = await response.json()
                    except:
                        body_data = await response.text()
                    
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "body": body_data,
                    }
        except Exception as e:
            return {
                "status": 0,
                "error": str(e),
                "error_type": type(e).__name__,
            }
    
    @tool(description="Load and parse a compliance test specification from a markdown file")
    async def load_test(self, path: str) -> Dict[str, Any]:
        """
        Load and parse a compliance test specification.
        
        Args:
            path: Path to the markdown test file
            
        Returns:
            Parsed test specification with setup and test cases
        """
        from .parser import TestParser
        
        test_path = Path(path)
        if not test_path.exists():
            return {"error": f"Test file not found: {path}"}
        
        parser = TestParser()
        return parser.parse(test_path)
    
    @tool(description="Validate a response against a natural language assertion using AI reasoning")
    async def validate_assertion(
        self,
        response: Dict[str, Any],
        assertion: str,
    ) -> Dict[str, Any]:
        """
        Validate a response against a natural language assertion.
        
        Uses AI reasoning to determine if the response meets the
        intent of the assertion.
        
        Args:
            response: The HTTP response to validate
            assertion: Natural language assertion (e.g., "Response contains a greeting")
            
        Returns:
            Validation result with passed, reasoning, and confidence
        """
        # This is the key method that uses LLM reasoning
        # The agent calling this tool will use its own judgment
        # to determine if the assertion passes
        
        return {
            "assertion": assertion,
            "response_summary": self._summarize_response(response),
            "instruction": (
                "Evaluate whether the response satisfies this assertion. "
                "Consider the intent, not just exact matching. "
                "Return your assessment as passed: true/false with reasoning."
            ),
        }
    
    @tool(description="Validate a response against strict deterministic assertions")
    async def validate_strict(
        self,
        response: Dict[str, Any],
        assertions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a response against strict (deterministic) assertions.
        
        Args:
            response: The HTTP response to validate
            assertions: Strict assertion rules (YAML-like dict)
            
        Returns:
            Validation result with passed and details for each assertion
        """
        from .validator import StrictValidator
        
        validator = StrictValidator()
        return validator.validate(response, assertions)
    
    @tool(description="Report the result of a test case")
    async def report_result(
        self,
        test_name: str,
        case_name: str,
        passed: bool,
        details: str,
        assertions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Report the result of a test case.
        
        Args:
            test_name: Name of the test suite
            case_name: Name of the specific test case
            passed: Whether the test passed
            details: Details about the result
            assertions: Individual assertion results
            
        Returns:
            Confirmation of result recording
        """
        result = TestResult(
            name=case_name,
            passed=passed,
            assertions=assertions or [],
            details=details,
            duration_ms=0,  # Would be tracked by executor
        )
        
        # Find or create suite result
        suite = next(
            (s for s in self.results if s.name == test_name),
            None
        )
        if not suite:
            suite = TestSuiteResult(
                name=test_name,
                test_file="",
                cases=[],
                passed=0,
                failed=0,
                skipped=0,
            )
            self.results.append(suite)
        
        suite.cases.append(result)
        if passed:
            suite.passed += 1
        else:
            suite.failed += 1
        
        return {
            "recorded": True,
            "test": test_name,
            "case": case_name,
            "passed": passed,
        }
    
    @tool(description="Get a summary of all test results so far")
    async def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all test results.
        
        Returns:
            Summary with total, passed, failed counts and details
        """
        total_passed = sum(s.passed for s in self.results)
        total_failed = sum(s.failed for s in self.results)
        total_skipped = sum(s.skipped for s in self.results)
        
        return {
            "total": total_passed + total_failed + total_skipped,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "suites": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "failed": s.failed,
                    "cases": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "details": c.details,
                        }
                        for c in s.cases
                    ],
                }
                for s in self.results
            ],
        }
    
    def _summarize_response(self, response: Dict[str, Any]) -> str:
        """Create a summary of a response for assertion validation."""
        status = response.get("status", "unknown")
        body = response.get("body", {})
        
        if isinstance(body, list):
            return f"Status: {status}, SSE stream with {len(body)} chunks"
        elif isinstance(body, dict):
            if "choices" in body:
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                return f"Status: {status}, Assistant response: {content[:200]}..."
            elif "error" in body:
                return f"Status: {status}, Error: {body.get('error')}"
            else:
                return f"Status: {status}, Body: {json.dumps(body)[:200]}..."
        else:
            return f"Status: {status}, Body: {str(body)[:200]}..."
