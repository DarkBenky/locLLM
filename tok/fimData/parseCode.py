from typing import List
from dataclasses import dataclass
import re
from hashlib import sha256
from tree_sitter_language_pack import get_parser

from dedup import compute_norm_hash

@dataclass
class ParseResult:
    code: str
    lang: str
    _hash: str
    embedding: object

    def __init__(self, code, lang):
        self.code = code
        self.lang = lang
        self._hash = sha256(code.encode()).hexdigest()
        self.norm_hash = compute_norm_hash(code, lang, TS_LANG_NAMES.get(lang))
        self.embedding = None

    def __hash__(self):
        return int(self._hash, 16)

@dataclass
class SampleUnparsed:
    codeFileContent: str
    lang: str

@dataclass
class SampleFile:
    path: str
    lang: str

LANG_PATTERNS = {
    "python": r"^(def |class |async def )",
    "javascript": r"^(function |class |const \w+\s*=\s*\(|export )",
    "typescript": r"^(function |class |const \w+\s*=\s*\(|export )",
    "go": r"^(func |type )",
    "java": r"^(\s*(public|private|protected|static).*\)\s*\{?$|class )",
    "c": r"^\w[\w\s\*]*\([^;]*\)\s*\{?$",
    "cpp": r"^\w[\w\s\*:<>]*\([^;]*\)\s*\{?$",
    "rust": r"^(fn |pub fn |struct |impl |enum )",
    "ruby": r"^(def |class |module )",
    "php": r"^(function |class |interface |trait )",
    "csharp": r"^(\s*(public|private|protected|internal).*\)\s*\{?$|class |interface |struct |enum )",
    "swift": r"^(func |class |struct |enum |protocol |extension )",
    "kotlin": r"^(fun |class |interface |object |data class )",
    "bash": r"^(\w+\s*\(\)\s*\{|function )",
    "glsl": r"^\w[\w\s\*]*\([^;]*\)\s*\{?$",
    "hlsl": r"^\w[\w\s\*]*\([^;]*\)\s*\{?$",
    "wgsl": r"^(fn |struct )",
    "opencl": r"^\w[\w\s\*]*\([^;]*\)\s*\{?$",
    "cuda": r"^\w[\w\s\*:<>]*\([^;]*\)\s*\{?$",
    "zig": r"^(fn |pub fn |const \w+\s*=\s*(struct|enum|union))",
    "odin": r"^\w+\s*::\s*(proc|struct|enum|union)",
    "lua": r"^(function |local function )",
    "scala": r"^(def |class |object |trait |case class )",
    "perl": r"^(sub |package )",
    "objc": r"^(@interface |@implementation |@protocol )",
    "nim": r"^(proc |func |type )",
    "tsx": r"^(function |class |const \w+\s*=\s*\(|export |interface )",
    "solidity": r"^(contract |interface |library |abstract contract )",
    "fortran": r"^(function |subroutine |module |program )",
    "erlang": r"^(\w+\(.*\)\s*->|-\w+)",
    "julia": r"^(function |struct |module |macro |abstract type |primitive type )",
    "crystal": r"^(def |class |module |abstract class |struct )",
    "ocaml": r"^(let |module |type )",
    "haxe": r"^(class |typedef |enum |interface )",
    "gleam": r"^(pub fn |fn |pub type |type )",
    "verilog": r"^(module |function |task |interface )",
    "systemverilog": r"^(module |class |interface |package )",
    "dart": r"^(class |\w[\w\s\*:<>]*\([^;]*\)\s*\{?$)",
    "mojo": r"^(fn |struct )",
    "r": r"^\w+\s*<-\s*function",
}

TS_LANG_NAMES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "csharp": "csharp",
    "swift": "swift",
    "kotlin": "kotlin",
    "bash": "bash",
    "glsl": "glsl",
    "hlsl": "hlsl",
    "wgsl": "wgsl",
    "opencl": "c",
    "cuda": "cuda",
    "zig": "zig",
    "odin": "odin",
    "lua": "lua",
    "scala": "scala",
    "perl": "perl",
    "objc": "objc",
    "tsx": "tsx",
    "solidity": "solidity",
    "fortran": "fortran",
    "erlang": "erlang",
    "julia": "julia",
    "crystal": "crystal",
    "ocaml": "ocaml",
    "haxe": "haxe",
    "gleam": "gleam",
    "verilog": "verilog",
    "systemverilog": "systemverilog",
    "dart": "dart",
}

TS_TOP_LEVEL_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "lexical_declaration", "export_statement"},
    "typescript": {"function_declaration", "class_declaration", "lexical_declaration", "export_statement", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"class_declaration", "interface_declaration", "method_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "rust": {"function_item", "struct_item", "impl_item", "enum_item"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "php": {"function_definition", "class_declaration", "interface_declaration", "trait_declaration", "enum_declaration"},
    "csharp": {"class_declaration", "interface_declaration", "struct_declaration", "enum_declaration", "record_declaration"},
    "swift": {"function_declaration", "class_declaration", "protocol_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "bash": {"function_definition"},
    "glsl": {"function_definition", "struct_specifier"},
    "hlsl": {"function_definition", "struct_specifier"},
    "wgsl": {"function_declaration", "struct_declaration"},
    "opencl": {"function_definition", "struct_specifier"},
    "cuda": {"function_definition", "struct_specifier"},
    "zig": {"Decl"},
    "odin": {"procedure_declaration", "struct_declaration"},
    "lua": {"function_declaration"},
    "scala": {"function_definition", "class_definition", "object_definition", "trait_definition"},
    "perl": {"subroutine_declaration_statement", "package_statement"},
    "objc": {"class_interface", "class_implementation", "protocol_declaration"},
    "tsx": {"export_statement", "function_declaration", "class_declaration", "interface_declaration", "lexical_declaration"},
    "solidity": {"contract_declaration", "interface_declaration", "library_declaration"},
    "fortran": {"function", "subroutine", "module", "program"},
    "erlang": {"fun_decl", "module_attribute", "export_attribute", "pp_define", "record_decl"},
    "julia": {"function_definition", "macro_definition", "module_definition", "struct_definition"},
    "crystal": {"method_def", "class_def", "module_def"},
    "ocaml": {"value_definition", "module_definition", "type_definition"},
    "haxe": {"class_declaration", "enum_declaration", "typedef_declaration", "interface_declaration"},
    "gleam": {"function", "type_definition"},
    "verilog": {"module_declaration", "package_or_generate_item_declaration"},
    "systemverilog": {"class_declaration", "interface_declaration", "module_declaration", "package_declaration"},
    "dart": {"class_definition", "function_signature"},
}

def splitByTopLevelRegex(code: str, lang: str) -> List[str]:
    pattern = LANG_PATTERNS.get(lang)
    if not pattern:
        return [code]
    lines = code.splitlines()
    regex = re.compile(pattern)
    indices = [i for i, line in enumerate(lines) if regex.match(line)]
    if not indices:
        return [code]
    indices.append(len(lines))
    chunks = []
    for start, end in zip(indices, indices[1:]):
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def splitByTopLevelTreeSitter(code: str, lang: str) -> List[str] | None:
    ts_name = TS_LANG_NAMES.get(lang)
    if not ts_name:
        return None
    try:
        parser = get_parser(ts_name)
    except Exception:
        return None

    tree = parser.parse(code.encode())
    root = tree.root_node

    if root.has_error:
        return None

    wanted_types = TS_TOP_LEVEL_TYPES.get(lang, set())
    chunks = []
    for node in root.children:
        if node.type in wanted_types:
            end_byte = node.end_byte
            if node.type == "function_signature":
                nxt = node.next_named_sibling
                if nxt is not None and nxt.type == "function_body":
                    end_byte = nxt.end_byte
            chunk = code[node.start_byte:end_byte].strip()
            if chunk:
                chunks.append(chunk)

    return chunks if chunks else None

def splitByTopLevel(code: str, lang: str) -> List[str]:
    chunks = splitByTopLevelTreeSitter(code, lang)
    if chunks is not None:
        return chunks
    return splitByTopLevelRegex(code, lang)

def parseCodeSample(sample: SampleUnparsed) -> List[ParseResult] | ParseResult:
    lang = sample.lang.strip().lower()
    code = sample.codeFileContent

    chunks = splitByTopLevel(code, lang)

    if len(chunks) == 1:
        return ParseResult(chunks[0], lang)

    parsed = [ParseResult(chunk, lang) for chunk in chunks]
    return parsed

def parseCodeFile(sample: SampleFile) -> List[ParseResult] | ParseResult:
    with open(sample.path, "r", encoding="utf-8") as f:
        content = f.read()
    return parseCodeSample(SampleUnparsed(content, sample.lang))

if __name__ == "__main__":
    testFiles = [
        SampleFile("./tests/test.py", "python"),
        SampleFile("./tests/test.js", "javascript"),
        SampleFile("./tests/test.ts", "typescript"),
        SampleFile("./tests/test.go", "go"),
        SampleFile("./tests/test.java", "java"),
        SampleFile("./tests/test.c", "c"),
        SampleFile("./tests/test.cpp", "cpp"),
        SampleFile("./tests/test.rs", "rust"),
        SampleFile("./tests/test.rb", "ruby"),
        SampleFile("./tests/test.php", "php"),
        SampleFile("./tests/test.cs", "csharp"),
        SampleFile("./tests/test.swift", "swift"),
        SampleFile("./tests/test.kt", "kotlin"),
        SampleFile("./tests/test.sh", "bash"),
        SampleFile("./tests/test.glsl", "glsl"),
        SampleFile("./tests/test.hlsl", "hlsl"),
        SampleFile("./tests/test.wgsl", "wgsl"),
        SampleFile("./tests/test.cl", "opencl"),
        SampleFile("./tests/test.cu", "cuda"),
        SampleFile("./tests/test.zig", "zig"),
        SampleFile("./tests/test.odin", "odin"),
        SampleFile("./tests/test.lua", "lua"),
        SampleFile("./tests/test.scala", "scala"),
        SampleFile("./tests/test.pl", "perl"),
        SampleFile("./tests/test.m", "objc"),
        SampleFile("./tests/test.nim", "nim"),
        SampleFile("./tests/test.tsx", "tsx"),
        SampleFile("./tests/test.sol", "solidity"),
        SampleFile("./tests/test.f90", "fortran"),
        SampleFile("./tests/test.erl", "erlang"),
        SampleFile("./tests/test.jl", "julia"),
        SampleFile("./tests/test.cr", "crystal"),
        SampleFile("./tests/test.ml", "ocaml"),
        SampleFile("./tests/test.hx", "haxe"),
        SampleFile("./tests/test.gleam", "gleam"),
        SampleFile("./tests/test.v", "verilog"),
        SampleFile("./tests/test.sv", "systemverilog"),
        SampleFile("./tests/test.dart", "dart"),
        SampleFile("./tests/test.mojo", "mojo"),
        SampleFile("./tests/test.R", "r"),
    ]

    for sample in testFiles:
        result = parseCodeFile(sample)
        results = result if isinstance(result, list) else [result]
        print(f"{sample.lang}: {len(results)} chunk(s)")
        for r in results:
            print(f"  hash={r._hash[:8]} len={len(r.code)}")