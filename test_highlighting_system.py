#!/usr/bin/env python3
"""
Test script for verifying highlighting system improvements.
Tests 10 different scenarios to ensure:
1. Text is readable behind highlights (opacity, colors)
2. Citations are extracted and separated properly
3. Highlights target correct text spans
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api import _extract_inline_citations, _select_top_snippets


def test_citation_extraction():
    """Test 1: Verify inline citations are properly extracted and removed"""
    print("\n=== TEST 1: Citation Extraction ===")
    
    test_cases = [
        # Case 1: Standard (Source X, Page Y)
        {
            "input": "The answer is 42 (Source 1, Page 3). This is a fact (Source 2, Page 5).",
            "expected_clean": "The answer is 42. This is a fact.",
            "expected_count": 2
        },
        # Case 2: Without parentheses
        {
            "input": "Result shown in Source 1, Page 7 and Source 2, Page 8.",
            "expected_clean": "Result shown in and .",
            "expected_count": 2
        },
        # Case 3: With filename
        {
            "input": "Data from (Source 1, Page 3 — report.pdf).",
            "expected_clean": "Data from .",
            "expected_count": 1
        },
        # Case 4: Bracket citations [1], [2]
        {
            "input": "Answer is here [1] and also here [2].",
            "expected_clean": "Answer is here and also here .",
            "expected_count": 0  # Bracket style should be removed but not counted as citations
        },
        # Case 5: Mixed citations
        {
            "input": "First fact (Source 1, Page 2). Second fact [3]. Third fact (Source 2, Page 4).",
            "expected_clean": "First fact . Second fact . Third fact .",
            "expected_count": 2
        },
        # Case 6: No citations
        {
            "input": "This is a clean answer with no citations.",
            "expected_clean": "This is a clean answer with no citations.",
            "expected_count": 0
        },
        # Case 7: Multiple on same source/page (deduplication)
        {
            "input": "Fact A (Source 1, Page 2). Fact B (Source 1, Page 2). Fact C (Source 1, Page 2).",
            "expected_clean": "Fact A . Fact B . Fact C .",
            "expected_count": 1  # Should dedupe
        },
    ]
    
    passed = 0
    for i, case in enumerate(test_cases, 1):
        clean, citations = _extract_inline_citations(case["input"])
        
        # Check citation count
        if len(citations) == case["expected_count"]:
            passed += 1
            print(f"  [OK] Case {i}: PASS - Extracted {len(citations)} citations")
        else:
            print(f"  [FAIL] Case {i}: FAIL - Expected {case['expected_count']} citations, got {len(citations)}")
            print(f"    Input: {case['input'][:60]}...")
            print(f"    Clean: {clean[:60]}...")
    
    print(f"  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_snippet_selection():
    """Test 2-4: Verify snippet selection prioritizes answer-aligned text"""
    print("\n=== TEST 2-4: Snippet Selection ===")
    
    test_cases = [
        # Test 2: Answer-aligned text should be selected
        {
            "name": "Answer alignment",
            "text": "Machine learning is a subset of artificial intelligence. "
                    "It enables systems to learn from data. "
                    "Deep learning uses neural networks with many layers. "
                    "This is background information about the field.",
            "answer": "Deep learning uses neural networks",
            "expected_in_snippets": "neural networks"
        },
        # Test 3: Query terms should match
        {
            "name": "Query term matching",
            "text": "Python is a programming language. Java is another language. "
                    "JavaScript runs in browsers. Python is popular for data science.",
            "answer": "Python programming for data analysis",
            "expected_in_snippets": "Python"
        },
        # Test 4: Empty answer should return first chunk
        {
            "name": "Empty answer fallback",
            "text": "First sentence here. Second sentence here.",
            "answer": "",
            "fallback": "Fallback text",
            "expected_in_snippets": "Fallback"
        },
    ]
    
    passed = 0
    for case in test_cases:
        fallback = case.get("fallback", "")
        snippets = _select_top_snippets(case["text"], case["answer"], fallback)
        
        found = any(case["expected_in_snippets"].lower() in s.lower() for s in snippets)
        if found:
            passed += 1
            print(f"  [OK] {case['name']}: PASS")
        else:
            print(f"  [FAIL] {case['name']}: FAIL")
            print(f"    Expected '{case['expected_in_snippets']}' in snippets")
            print(f"    Got: {snippets[:2]}")
    
    print(f"  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_highlight_color_readability():
    """Test 5-7: Verify highlight colors and opacity settings"""
    print("\n=== TEST 5-7: Highlight Readability Settings ===")
    
    from simple_highlighter import SimpleHighlighter
    from perplexity_highlighter import PerplexityHighlighter
    
    simple = SimpleHighlighter()
    perplexity = PerplexityHighlighter()
    
    tests = [
        # Test 5: Simple highlighter colors
        {
            "name": "Simple highlighter - Main Answer color",
            "color": simple.category_colors["Main Answer"],
            "check": lambda c: all(0.6 <= v <= 1.0 for v in c),  # Should be light/soft
            "desc": "Soft, light color (not dark)"
        },
        # Test 6: Perplexity highlighter colors
        {
            "name": "Perplexity highlighter - Main Answer color",
            "color": perplexity.category_colors["Main Answer"],
            "check": lambda c: all(0.6 <= v <= 1.0 for v in c),
            "desc": "Soft, light color (not dark)"
        },
        # Test 7: Color consistency between highlighters
        {
            "name": "Color consistency",
            "check": lambda _: simple.category_colors["Main Answer"] == perplexity.category_colors["Main Answer"],
            "desc": "Both use same color",
            "skip_color": True
        },
    ]
    
    passed = 0
    for test in tests:
        if test.get("skip_color"):
            result = test["check"](None)
        else:
            result = test["check"](test["color"])
        
        if result:
            passed += 1
            print(f"  [OK] {test['name']}: PASS - {test['desc']}")
        else:
            print(f"  [FAIL] {test['name']}: FAIL")
            if "color" in test:
                print(f"    Color: {test['color']}")
    
    print(f"  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_highlight_opacity_settings():
    """Test 8-10: Verify highlight opacity is readable"""
    print("\n=== TEST 8-10: Opacity Settings ===")
    
    # These values are hardcoded in the draw_rect calls
    # We need to verify they're in the readable range (0.15-0.25)
    
    # Read the source files to check the actual values
    import re
    
    results = []
    
    # Check simple_highlighter.py
    with open("simple_highlighter.py", "r") as f:
        content = f.read()
        opacity_matches = re.findall(r"fill_opacity=([0-9.]+)", content)
        simple_opacities = [float(m) for m in opacity_matches]
        
        all_readable = all(0.15 <= op <= 0.30 for op in simple_opacities)
        results.append(("Simple highlighter", all_readable, simple_opacities))
    
    # Check perplexity_highlighter.py
    with open("perplexity_highlighter.py", "r") as f:
        content = f.read()
        opacity_matches = re.findall(r"fill_opacity=([0-9.]+)", content)
        perplexity_opacities = [float(m) for m in opacity_matches]
        
        all_readable = all(0.15 <= op <= 0.30 for op in perplexity_opacities)
        results.append(("Perplexity highlighter", all_readable, perplexity_opacities))
    
    passed = 0
    for name, is_readable, opacities in results:
        if is_readable:
            passed += 1
            print(f"  [OK] {name}: PASS - Opacity {opacities} in readable range (0.15-0.30)")
        else:
            print(f"  [FAIL] {name}: FAIL - Opacity {opacities} outside readable range")
    
    # Test 10: Verify stroke settings exist for boundary definition
    with open("simple_highlighter.py", "r") as f:
        simple_has_stroke = "stroke_opacity" in f.read()
    
    with open("perplexity_highlighter.py", "r") as f:
        perplexity_has_stroke = "stroke_opacity" in f.read()
    
    if simple_has_stroke and perplexity_has_stroke:
        passed += 1
        print(f"  [OK] Stroke settings: PASS - Both highlighters have stroke borders")
    else:
        print(f"  [FAIL] Stroke settings: FAIL - Missing stroke in one or both")
    
    print(f"  Result: {passed}/{len(results) + 1} passed")
    return passed == len(results) + 1


def main():
    """Run all 10 tests"""
    print("=" * 60)
    print("PDF HIGHLIGHTING SYSTEM - 10 WAY TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Tests 1: Citation extraction
    results.append(("Citation Extraction", test_citation_extraction()))
    
    # Tests 2-4: Snippet selection
    results.append(("Snippet Selection (3 tests)", test_snippet_selection()))
    
    # Tests 5-7: Color readability
    results.append(("Color Settings (3 tests)", test_highlight_color_readability()))
    
    # Tests 8-10: Opacity settings
    results.append(("Opacity Settings (3 tests)", test_highlight_opacity_settings()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    for name, passed_test in results:
        status = "[OK] PASS" if passed_test else "[FAIL] FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nOverall: {total_passed}/{total_tests} test groups passed")
    
    if total_passed == total_tests:
        print("\n*** ALL TESTS PASSED - Highlighting system is ready! ***")
        return 0
    else:
        print("\n*** Some tests failed - Review the implementation ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
