"""
Strict assertion validator.

Validates responses against deterministic YAML assertions.
"""

import re
import json
from typing import Dict, Any, List, Union


class StrictValidator:
    """
    Validates responses against strict (deterministic) assertions.
    
    Supports:
    - Exact matching
    - Regex matching (prefix with /)
    - Type checking (type(string), type(number), etc.)
    - Existence checks (exists, not_exists)
    - Length checks (length(N))
    - Contains checks (contains("str"))
    - JSONPath-like access (body.choices[0].message.content)
    """
    
    def validate(
        self,
        response: Dict[str, Any],
        assertions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate response against strict assertions.
        
        Args:
            response: HTTP response dict with status, headers, body
            assertions: Assertion rules
            
        Returns:
            Validation result with passed flag and details
        """
        results = []
        all_passed = True
        
        for key, expected in assertions.items():
            if key == "status":
                result = self._check_status(response.get("status"), expected)
            elif key == "headers":
                result = self._check_headers(response.get("headers", {}), expected)
            elif key == "body":
                result = self._check_body(response.get("body", {}), expected)
            elif key == "format":
                result = self._check_format(response, expected)
            elif key == "chunks":
                result = self._check_chunks(response.get("body", []), expected)
            elif key == "final_chunk":
                result = self._check_final_chunk(response.get("body", []), expected)
            elif key == "events":
                result = self._check_events(response.get("events", []), expected)
            else:
                result = {"path": key, "passed": False, "reason": f"Unknown assertion key: {key}"}
            
            results.append(result)
            if not result.get("passed", False):
                all_passed = False
        
        return {
            "passed": all_passed,
            "results": results,
        }
    
    def _check_status(self, actual: Any, expected: Any) -> Dict[str, Any]:
        """Check HTTP status code."""
        if isinstance(expected, list):
            passed = actual in expected
            reason = f"Status {actual} in {expected}" if passed else f"Status {actual} not in {expected}"
        else:
            passed = actual == expected
            reason = f"Status {actual} == {expected}" if passed else f"Status {actual} != {expected}"
        
        return {"path": "status", "passed": passed, "reason": reason}
    
    def _check_headers(self, actual: Dict[str, str], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Check response headers."""
        results = []
        all_passed = True
        
        for header, value in expected.items():
            actual_value = actual.get(header) or actual.get(header.lower())
            result = self._check_value(f"headers.{header}", actual_value, value)
            results.append(result)
            if not result["passed"]:
                all_passed = False
        
        return {
            "path": "headers",
            "passed": all_passed,
            "details": results,
        }
    
    def _check_body(self, actual: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Check response body using JSONPath-like assertions."""
        results = []
        all_passed = True
        
        for path, value in expected.items():
            actual_value = self._get_path(actual, path)
            result = self._check_value(f"body.{path}", actual_value, value)
            results.append(result)
            if not result["passed"]:
                all_passed = False
        
        return {
            "path": "body",
            "passed": all_passed,
            "details": results,
        }
    
    def _check_format(self, response: Dict[str, Any], expected: str) -> Dict[str, Any]:
        """Check response format (e.g., sse)."""
        actual = response.get("format")
        passed = actual == expected
        return {
            "path": "format",
            "passed": passed,
            "reason": f"Format {actual} == {expected}" if passed else f"Format {actual} != {expected}",
        }
    
    def _check_chunks(self, chunks: List[Any], expected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check SSE chunks."""
        if not isinstance(chunks, list):
            return {"path": "chunks", "passed": False, "reason": "Response is not SSE chunks"}
        
        results = []
        all_passed = True
        
        for assertion in expected:
            # Check if any chunk matches the assertion
            matched = False
            for chunk in chunks:
                if self._chunk_matches(chunk, assertion):
                    matched = True
                    break
            
            if not matched:
                all_passed = False
                results.append({
                    "assertion": assertion,
                    "passed": False,
                    "reason": "No chunk matched this assertion",
                })
            else:
                results.append({
                    "assertion": assertion,
                    "passed": True,
                })
        
        return {
            "path": "chunks",
            "passed": all_passed,
            "details": results,
        }
    
    def _check_final_chunk(self, chunks: List[Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Check the final chunk of an SSE stream."""
        if not chunks:
            return {"path": "final_chunk", "passed": False, "reason": "No chunks"}
        
        final = chunks[-1]
        results = []
        all_passed = True
        
        for path, value in expected.items():
            actual_value = self._get_path(final, path)
            result = self._check_value(f"final_chunk.{path}", actual_value, value)
            results.append(result)
            if not result["passed"]:
                all_passed = False
        
        return {
            "path": "final_chunk",
            "passed": all_passed,
            "details": results,
        }
    
    def _check_events(self, events: List[Any], expected: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check events (for multi-agent tests)."""
        results = []
        all_passed = True
        
        for exp_event in expected:
            matched = False
            for event in events:
                if self._event_matches(event, exp_event):
                    matched = True
                    break
            
            if not matched:
                all_passed = False
                results.append({
                    "expected": exp_event,
                    "passed": False,
                    "reason": "Event not found",
                })
            else:
                results.append({
                    "expected": exp_event,
                    "passed": True,
                })
        
        return {
            "path": "events",
            "passed": all_passed,
            "details": results,
        }
    
    def _chunk_matches(self, chunk: Dict[str, Any], assertion: Dict[str, Any]) -> bool:
        """Check if a chunk matches an assertion."""
        for path, value in assertion.items():
            actual = self._get_path(chunk, path)
            result = self._check_value(path, actual, value)
            if not result["passed"]:
                return False
        return True
    
    def _event_matches(self, event: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check if an event matches expected criteria."""
        for key, value in expected.items():
            if key == "type":
                if event.get("type") != value:
                    return False
            else:
                actual = event.get(key)
                result = self._check_value(key, actual, value)
                if not result["passed"]:
                    return False
        return True
    
    def _get_path(self, obj: Any, path: str) -> Any:
        """Get value at JSONPath-like path."""
        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]
        
        current = obj
        for part in parts:
            if current is None:
                return None
            
            if part == "*":
                # Wildcard - return all values
                if isinstance(current, list):
                    return [self._get_path(item, ".".join(parts[parts.index(part)+1:])) for item in current]
                return None
            
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
        
        return current
    
    def _check_value(self, path: str, actual: Any, expected: Any) -> Dict[str, Any]:
        """Check a single value against expected."""
        # Handle special assertions
        if isinstance(expected, str):
            # Regex
            if expected.startswith("/") and expected.endswith("/"):
                pattern = expected[1:-1]
                if actual is None:
                    return {"path": path, "passed": False, "reason": f"Value is None, expected regex {pattern}"}
                passed = bool(re.search(pattern, str(actual)))
                return {
                    "path": path,
                    "passed": passed,
                    "reason": f"Regex {'matched' if passed else 'did not match'}",
                }
            
            # Type check
            if expected.startswith("type(") and expected.endswith(")"):
                expected_type = expected[5:-1]
                type_map = {
                    "string": str,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                    "null": type(None),
                }
                expected_py_type = type_map.get(expected_type)
                passed = isinstance(actual, expected_py_type) if expected_py_type else False
                return {
                    "path": path,
                    "passed": passed,
                    "reason": f"Type is {type(actual).__name__}, expected {expected_type}",
                }
            
            # Exists check
            if expected == "exists":
                passed = actual is not None
                return {"path": path, "passed": passed, "reason": "Value exists" if passed else "Value does not exist"}
            
            if expected == "not_exists":
                passed = actual is None
                return {"path": path, "passed": passed, "reason": "Value does not exist" if passed else "Value exists"}
            
            if expected == "not_null":
                passed = actual is not None
                return {"path": path, "passed": passed, "reason": "Value is not null" if passed else "Value is null"}
            
            # Length check
            if expected.startswith("length(") and expected.endswith(")"):
                expected_len = int(expected[7:-1])
                actual_len = len(actual) if hasattr(actual, "__len__") else 0
                passed = actual_len == expected_len
                return {
                    "path": path,
                    "passed": passed,
                    "reason": f"Length is {actual_len}, expected {expected_len}",
                }
            
            # Contains check
            if expected.startswith("contains(") and expected.endswith(")"):
                substring = expected[9:-1].strip('"\'')
                passed = substring in str(actual) if actual else False
                return {
                    "path": path,
                    "passed": passed,
                    "reason": f"Contains '{substring}': {passed}",
                }
        
        # List of acceptable values
        if isinstance(expected, list):
            passed = actual in expected
            return {
                "path": path,
                "passed": passed,
                "reason": f"Value {actual} in {expected}" if passed else f"Value {actual} not in {expected}",
            }
        
        # Exact match
        passed = actual == expected
        return {
            "path": path,
            "passed": passed,
            "reason": f"Value {actual} == {expected}" if passed else f"Value {actual} != {expected}",
        }
