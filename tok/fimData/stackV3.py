from datasets import load_dataset

from parseCode import SampleUnparsed, parseCodeSample

CACHE_DIR = "data/"

STACKV3_LANG_MAP = {
    "Python": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "C": "c",
    "C++": "cpp",
    "C#": "csharp",
    "Java": "java",
    "Go": "go",
    "Rust": "rust",
    "Ruby": "ruby",
    "PHP": "php",
    "Swift": "swift",
    "Kotlin": "kotlin",
    "Scala": "scala",
    "Lua": "lua",
    "Perl": "perl",
    "Objective-C": "objc",
    "Shell": "bash",
    "Cuda": "cuda",
    "GLSL": "glsl",
    "HLSL": "hlsl",
    "OpenCL": "opencl",
    "Zig": "zig",
    "Nim": "nim",
    "Odin": "odin",
    "WGSL": "wgsl",
    "TSX": "tsx",
    "Solidity": "solidity",
    "Fortran": "fortran",
    "Fortran Free Form": "fortran",
    "Erlang": "erlang",
    "Julia": "julia",
    "Crystal": "crystal",
    "OCaml": "ocaml",
    "Haxe": "haxe",
    "Gleam": "gleam",
    "Verilog": "verilog",
    "SystemVerilog": "systemverilog",
    "Dart": "dart",
    "R": "r",
    "Mojo": "mojo",
}

DEFAULT_SUPPORTED = set(STACKV3_LANG_MAP)


def stack_v3_gen(supported_langs=None):
    ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, cache_dir=CACHE_DIR)
    if isinstance(supported_langs, str):
        supported_langs = {s.strip().lower() for s in supported_langs.split(",")}
    elif supported_langs is not None:
        supported_langs = {s.strip().lower() for s in supported_langs}
    for repo in ds:
        try:
            for f in repo["files"]:
                sv3_lang = f["language"]
                parse_lang = STACKV3_LANG_MAP.get(sv3_lang)
                if parse_lang is None:
                    continue
                if supported_langs is not None and sv3_lang.lower() not in supported_langs:
                    continue
                yield {
                    "text": f["content"],
                    "lang": parse_lang,
                    "category": sv3_lang,
                }
        except:
            continue


def stack_v3_fim_gen(supported_langs=None):
    for sample in stack_v3_gen(supported_langs):
        try:
            result = parseCodeSample(SampleUnparsed(sample["text"], sample["lang"]))
            results = result if isinstance(result, list) else [result]
            for r in results:
                yield {
                    "text": r.code,
                    "lang": r.lang,
                    "category": sample["category"],
                    "hash": r._hash,
                    "embedding": None,
                }
        except:
            continue


if __name__ == "__main__":
    for i, sample in enumerate(stack_v3_fim_gen()):
        print(f"{sample['category']:12s} {sample['lang']:10s} {sample['hash'][:8]} {len(sample['text'])}")
        if i >= 10:
            break
