"""
Test executor for running compliance tests.

Orchestrates test execution using the TestRunnerSkill.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExecutionOptions:
    """Options for test execution."""
    base_url: str = "http://localhost:8765"
    cache_mode: str = "read_write"  # read_write, write_only, disabled
    temperature: float = 0.0
    timeout: int = 60
    strict_only: bool = False
    tags: Optional[List[str]] = None
    skip_tags: Optional[List[str]] = None
    parallel: int = 1


class TestExecutor:
    """
    Executes compliance tests against an SDK server.
    
    Can run in two modes:
    1. Agentic: Uses AI to validate natural language assertions
    2. Strict-only: Only runs deterministic assertions (faster, no LLM)
    """
    
    def __init__(self, options: ExecutionOptions):
        self.options = options
        self.results: List[Dict[str, Any]] = []
    
    async def run_file(self, path: Path) -> Dict[str, Any]:
        """
        Run all tests in a single file.
        
        Args:
            path: Path to test file
            
        Returns:
            Test results for the file
        """
        from .parser import TestParser
        from .skill import TestRunnerSkill
        from .validator import StrictValidator
        
        parser = TestParser()
        spec = parser.parse(path)
        
        # Check tags
        if self.options.tags:
            if not any(tag in spec.get("tags", []) for tag in self.options.tags):
                return {
                    "name": spec["name"],
                    "skipped": True,
                    "reason": "No matching tags",
                }
        
        if self.options.skip_tags:
            if any(tag in spec.get("tags", []) for tag in self.options.skip_tags):
                return {
                    "name": spec["name"],
                    "skipped": True,
                    "reason": "Skipped tag",
                }
        
        skill = TestRunnerSkill(
            base_url=self.options.base_url,
            timeout=self.options.timeout,
        )
        
        try:
            results = []
            
            for case in spec.get("cases", []):
                start_time = time.time()
                
                case_result = await self._run_case(skill, spec, case)
                
                case_result["duration_ms"] = (time.time() - start_time) * 1000
                results.append(case_result)
            
            passed = sum(1 for r in results if r.get("passed", False))
            failed = sum(1 for r in results if not r.get("passed", False) and not r.get("skipped", False))
            skipped = sum(1 for r in results if r.get("skipped", False))
            
            return {
                "name": spec["name"],
                "file": str(path),
                "cases": results,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
            }
        finally:
            await skill.shutdown()
    
    async def _run_case(
        self,
        skill: "TestRunnerSkill",
        spec: Dict[str, Any],
        case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single test case."""
        from .validator import StrictValidator
        
        case_name = case.get("name", "unnamed")
        request = case.get("request")
        
        if not request:
            return {
                "name": case_name,
                "skipped": True,
                "reason": "No request defined",
            }
        
        # Make request
        response = await skill.http_request(
            method=request.get("method", "POST"),
            path=request.get("path", "/chat/completions"),
            body=request.get("body"),
            headers=request.get("headers"),
            stream=request.get("stream", False),
        )
        
        # Handle request error
        if response.get("error"):
            if case.get("expected") == "failure":
                return {
                    "name": case_name,
                    "passed": True,
                    "reason": "Expected failure occurred",
                }
            return {
                "name": case_name,
                "passed": False,
                "reason": f"Request error: {response.get('error')}",
            }
        
        assertion_results = []
        all_passed = True
        
        # Run strict assertions
        strict = case.get("strict")
        if strict:
            validator = StrictValidator()
            strict_result = validator.validate(response, strict)
            
            assertion_results.append({
                "type": "strict",
                "passed": strict_result["passed"],
                "details": strict_result["results"],
            })
            
            if not strict_result["passed"]:
                all_passed = False
        
        # Run natural language assertions (if not strict-only mode)
        if not self.options.strict_only:
            for assertion in case.get("assertions", []):
                # In agentic mode, we'd use LLM to validate
                # For now, we just record the assertion for the agent
                assertion_results.append({
                    "type": "natural",
                    "assertion": assertion,
                    "passed": None,  # To be validated by agent
                    "needs_validation": True,
                })
        
        # Check expected outcome
        expected = case.get("expected", "success")
        if expected == "failure":
            # For expected failures, check if we got an error response
            if response.get("status", 200) >= 400:
                all_passed = True
            else:
                all_passed = False
        
        return {
            "name": case_name,
            "passed": all_passed,
            "response": {
                "status": response.get("status"),
                "body_preview": str(response.get("body", ""))[:200],
            },
            "assertions": assertion_results,
        }
    
    async def run_directory(self, path: Path) -> Dict[str, Any]:
        """
        Run all tests in a directory.
        
        Args:
            path: Path to test directory
            
        Returns:
            Aggregated test results
        """
        test_files = list(path.glob("**/*.md"))
        
        results = []
        for test_file in test_files:
            result = await self.run_file(test_file)
            results.append(result)
        
        total_passed = sum(r.get("passed", 0) for r in results if not r.get("skipped"))
        total_failed = sum(r.get("failed", 0) for r in results if not r.get("skipped"))
        total_skipped = sum(1 for r in results if r.get("skipped"))
        
        return {
            "total": len(results),
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "suites": results,
        }
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print results to console."""
        print("\nCompliance Test Results")
        print("=" * 50)
        
        for suite in results.get("suites", []):
            if suite.get("skipped"):
                print(f"\n⊘ {suite['name']}: SKIPPED ({suite.get('reason', '')})")
                continue
            
            status = "✓" if suite.get("failed", 0) == 0 else "✗"
            print(f"\n{status} {suite['name']}: {suite.get('passed', 0)}/{suite.get('passed', 0) + suite.get('failed', 0)} passed")
            
            for case in suite.get("cases", []):
                case_status = "✓" if case.get("passed") else "✗"
                print(f"  {case_status} {case.get('name', 'unnamed')}")
                
                if not case.get("passed") and case.get("assertions"):
                    for assertion in case["assertions"]:
                        if not assertion.get("passed", True):
                            print(f"    → {assertion.get('details', assertion.get('assertion', ''))}")
        
        print("\n" + "=" * 50)
        print(f"Summary: {results.get('passed', 0)}/{results.get('passed', 0) + results.get('failed', 0)} tests passed")
        if results.get("skipped"):
            print(f"         {results.get('skipped')} skipped")
