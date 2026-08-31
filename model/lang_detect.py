"""Lightweight code-language detection for the inference server.

Returns the locLLM `<lang>` tag string (as used in training — see
main_big.normalize_lang) or None when uncertain. Conservative: prefers strong,
language-specific markers over generic ones so we don't mis-tag plain text.
"""
from __future__ import annotations

import re

# Ordered: first match wins. Keys are the trained `<lang>` tag strings.
_RULES: list[tuple[str, re.Pattern]] = [
    ("shell", re.compile(r"#!\s*/(usr/bin/|bin/)?(env\s+)?(ba|z|k)?sh\b|^\s*(\$|echo\s+\S+|for\s+\w+\s+in\s+.*;\s*do)")),
    ("python", re.compile(r"^\s*(from\s+\w+\s+import\s+\w+|import\s+\w+|def\s+\w+\s*\(|class\s+\w+\s*[:\()]|async\s+def\s+)|@(?:staticmethod|classmethod|property)|self\.|print\(|elif\s+.*:|except\s+\w+\s*:|\b__init__\b")),
    ("c++",   re.compile(r"std::|namespace\s+\w+(::\w+)*\s*\{|using\s+namespace\s+|cout\s*<<|cin\s*>>|template\s*<|#include\s*<(iostream|vector|string|map|set|algorithm|queue|bits/stdc\+\+\.h)>")),
    ("c",     re.compile(r"#include\s*<stdio\.h>|#include\s*<stdlib\.h>|printf\(|scanf\(|int\s+main\s*\(|malloc\(|fprintf\(")),
    ("java",  re.compile(r"public\s+(?:final\s+)?(?:class|interface|enum)|System\.out\.print|import\s+java\.|new\s+\w+\s*\([^)]*\)\s*\{|@Override|public\s+static\s+void\s+main")),
    ("c#",    re.compile(r"using\s+System(\.\w+)*;|Console\.Write|namespace\s+\w+\s*\{|class\s+\w+\s*:\s*\w+|public\s+(?:partial\s+)?class\s+\w+")),
    ("go",    re.compile(r"^\s*package\s+main|func\s+\w+\s*\(|import\s*\(|go\s+func\b|err\s*:?=|\bchan\s+\w+")),
    ("rust",  re.compile(r"^\s*(use\s+\w+(::\w+)*;|fn\s+\w+\s*\(|let\s+(?:mut\s+)?\w+\s*[:=])|impl\s+\w+|match\s+\w+\s*\{|\bpub\s+fn\b")),
    ("typescript", re.compile(r":\s*(string|number|boolean|void|any)\b|interface\s+\w+\s*\{|type\s+\w+\s*=|import\s+.*\s+from\s+['\"]|as\s+const\b")),
    ("javascript", re.compile(r"function\s+\w+\s*\(|=>\s*\{|console\.log|require\(|const\s+\w+\s*=\s*(?:\([^)]*\)\s*=>|function)|export\s+default|document\.(getElementById|querySelector)")),
    ("ruby",  re.compile(r"^\s*(def\s+\w+.*\n.*end|require\s+['\"]\w+['\"]|puts\s+|attr_(accessor|reader|writer)|class\s+\w+\s*<\s*\w+)")),
    ("php",   re.compile(r"<\?php|\$this->|\$[a-zA-Z_]+\s*->|echo\s+\$|=>\s*function")),
    ("swift", re.compile(r"import\s+Swift|func\s+\w+\s*\([^)]*\)\s*(->\s*\w+)?\s*\{|guard\s+let\s+\w+|let\s+\w+\s*=\s*\w+\s*\?\?")),
    ("kotlin", re.compile(r"\bfun\s+\w+\s*\(|val\s+\w+\s*[:=]|var\s+\w+\s*[:=]|class\s+\w+\s*(:|\()")),
    ("sql",   re.compile(r"^\s*(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE|WITH\s+\w+\s+AS)\b", re.I)),
    ("html",  re.compile(r"<!DOCTYPE\s+html|<html|<head|<body|</(div|span|p|h[1-6])>")),
    ("css",   re.compile(r"(#[.\w-]+\s*\{|[.\w-]+\s*\{\s*(color|margin|padding|display|font-size)\s*:)")),
    # json is intentionally excluded: it trains as plain LM (no FIM lang tag).
    ("yaml",  re.compile(r"(?:^|\n)\s*[\w.-]+:\s*(?:\S.*)?(?:\n\s*[\w.-]+:\s*)")),
]


def infer_code_lang(text: str) -> str | None:
    """Return the best-guess `<lang>` tag string, or None if unknown."""
    if not text:
        return None
    # JSON/yaml are weak rules: only accept if they appear at the very start
    # (strong indicator of a data file, not a code file).
    head = text[:400]
    for lang, pat in _RULES:
        if lang == "json":
            if pat.match(head.lstrip()):
                return lang
            continue
        if pat.search(text[:6000]):
            return lang
    return None
