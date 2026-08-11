import ast
import re
import json
import sys
import random

from datasets import load_dataset

from uploader import DatasetPipeline

API = "http://91.98.145.193:8823/"
CACHE_DIR = "data/"
CHECKPOINT_FILE = "checkpoint.json"

COMMON_LANG = ["Python", "JavaScript", "C++", "Java", "C", "Go", "TypeScript", "Ruby", "Rust", "PHP", "Swift", "C#", "Kotlin", "Scala", "Dart", "Objective-C", "Perl", "Lua", "SQL", "HTML", "CSS", "JSON", "YAML", "Markdown", "XML"]

pipe = DatasetPipeline(api=API, checkpoint_file=CHECKPOINT_FILE)


def to_chatml(messages, tools=None):
    parts = []
    for m in messages:
        role = m["role"]

        if role == "system":
            body = m["content"]
            if tools:
                body += (
                    "\n\nYou have access to the following tools:\n"
                    f"<tools>\n{json.dumps(tools, ensure_ascii=False)}\n</tools>\n\n"
                    'To call a tool, respond with:\n<tool_call>\n'
                    '{"name": "...", "arguments": {...}}\n</tool_call>'
                )
            parts.append(f"<|im_start|>system\n{body}<|im_end|>")

        elif role == "assistant":
            body = m.get("content", "")
            if m.get("reasoning_content"):
                body = f"<think>\n{m['reasoning_content']}\n</think>\n\n{body}"
            if m.get("tool_calls"):
                calls = "\n".join(
                    "<tool_call>\n"
                    + json.dumps({
                        "name": c["function"]["name"],
                        "arguments": json.loads(c["function"]["arguments"])
                                     if isinstance(c["function"]["arguments"], str)
                                     else c["function"]["arguments"],
                    }, ensure_ascii=False)
                    + "\n</tool_call>"
                    for c in m["tool_calls"]
                )
                body = f"{body}\n{calls}" if body else calls
            parts.append(f"<|im_start|>assistant\n{body}<|im_end|>")

        elif role == "tool":
            parts.append(f"<|im_start|>tool\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>")

        else:
            parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")

    return "\n".join(parts)


def parse_codealchemy(text):
    messages = []
    parts = re.split(r"\n(?=(?:User|Assistant):\n)", text)
    for part in parts:
        m = re.match(r"^(User|Assistant):\n(.*)$", part, re.S)
        if not m:
            continue
        role = "user" if m.group(1) == "User" else "assistant"
        content = m.group(2).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


_swh_s3 = None
_swh_cache = {}
_swh_cache_max = 1024
_swh_missing_warned = False


def fetch_swh_source(blob_id):
    global _swh_s3, _swh_missing_warned
    if blob_id in _swh_cache:
        return _swh_cache[blob_id]
    try:
        import gzip
        import boto3
        from botocore.config import Config
        from botocore import UNSIGNED
    except ImportError:
        if not _swh_missing_warned:
            _swh_missing_warned = True
            print("[code-alchemy] boto3 not installed — code-dev/code-dialogue rows will be skipped (pip install boto3)")
        return None
    try:
        if _swh_s3 is None:
            _swh_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        obj = _swh_s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=obj["Body"]) as f:
            seed = f.read().decode("utf-8", errors="ignore")
        if len(_swh_cache) >= _swh_cache_max:
            _swh_cache.clear()
        _swh_cache[blob_id] = seed
        return seed
    except Exception as e:
        print(f"[code-alchemy] S3 fetch failed for {blob_id}: {e}")
        return None


@pipe.register("code_alchemy")
def code_alchemy_gen(configs=None):
    if configs is None:
        configs = ["code-enhance", "code-qa", "code-dev", "code-dialogue", "code-trace"]

    streams = []
    for c in configs:
        try:
            streams.append(iter(load_dataset("open-alchemy/code-alchemy", c, split="train", streaming=True, cache_dir=CACHE_DIR)))
        except Exception as e:
            print(f"[code-alchemy] skipping config {c}: {e}")

    if not streams:
        return

    active = list(range(len(streams)))
    while active:
        for i in active[:]:
            try:
                row = next(streams[i])
            except StopIteration:
                active.remove(i)
                continue
            try:
                text = row.get("text")
                if not text and row.get("text_with_placeholders"):
                    seed = fetch_swh_source(row["blob_id"])
                    if seed is None:
                        continue
                    text = row["text_with_placeholders"].replace("{{{REPLACE_WITH_BLOB_ID_SOURCE}}}", seed)
                if not text:
                    continue
                parsed = parse_codealchemy(text)
                if parsed:
                    text = to_chatml(parsed)
                if not text or not text.strip():
                    continue
                yield {"text": text, "category": "code_alchemy"}
            except:
                continue


@pipe.register("evo_code_instruction")
def evo_code_instruction_gen():
    ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            prompt = repo["instruction"]
            response = repo["output"]
            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": "instruction_code",
            }
        except:
            continue


@pipe.register("ultrachat")
def ultrachat_gen():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            msg = to_chatml(repo["messages"])
            yield {
                "text": msg,
                "category": "ultrachat",
            }
        except:
            continue


@pipe.register("distillation")
def distillation_qwen_kimi_glm_gen():
    ds = load_dataset("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation", "sft_balanced", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            msgs = repo["messages_json"]
            tools = repo.get("tools_json")
            msg = to_chatml(msgs, tools=tools)
            yield {
                "text": msg,
                "category": "instruction_code",
            }
        except:
            continue


@pipe.register("codealpha")
def codealpha_gen():
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            prompt = repo["instruction"]
            response = repo["response"]
            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": "instruction_code",
            }
        except:
            continue


@pipe.register("small_talk")
def small_talk_gen():
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            msg = to_chatml(repo["messages"])
            yield {
                "text": msg,
                "category": "small_talk",
            }
        except:
            continue


@pipe.register("code_security")
def code_security_gen():
    ds = load_dataset("ayshajavd/code-security-vulnerability-dataset", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            code = repo["code"]
            is_vulnerable = repo["is_vulnerable"]
            lang = repo["language"]
            prompt = f"Fix the following {lang} code:\n{code}\n"
            if is_vulnerable:
                code_fix = repo["code_fixed"]
                response = f"Fixed code:\n{code_fix}"
                msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
                yield {
                    "text": msg,
                    "category": f"{lang}_instruction_code",
                }
            else:
                response = f"The following {lang} code is not vulnerable"
                msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
                yield {
                    "text": msg,
                    "category": f"{lang}_instruction_code",
                }
        except:
            continue


@pipe.register("tiny_codes")
def tiny_codes_gen():
    ds = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            lang = repo["programming_language"]
            problem = repo["prompt"]
            response = repo["response"]
            msg = to_chatml([{"role": "user", "content": problem}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": f"{lang}_instruction_code",
            }
        except:
            continue


@pipe.register("code_gen")
def code_gen():
    ds = load_dataset("pengyunie/codesearchnet-codegen", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            problem = repo["problem"]
            output = repo["output"]
            lang = repo["language"]
            msg = to_chatml([{"role": "user", "content": problem}, {"role": "assistant", "content": output}])
            yield {
                "text": msg,
                "category": f"{lang}_instruction_code",
            }
        except:
            continue


@pipe.register("nemotron_codealpaca")
def nemotron_codealpaca_gen():
    ds = load_dataset("JessieWei/GLM-5.2-FP8-nemotron-codealpaca", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            conversation = repo["conversations"]
            msg = to_chatml(conversation)
            yield {
                "text": msg,
                "category": "instruction_code_alpaca",
            }
        except:
            continue


@pipe.register("code_feedback")
def code_feedback_gen():
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            prompt = repo["query"]
            response = repo["answer"]
            lang = repo["lang"]
            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": f"{lang}_instruction_code",
            }
        except:
            continue


@pipe.register("star_coder", prefetch=8)
def star_coder_gen():
    ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            content = repo["content"]
            yield {
                "text": content,
                "category": "star_coder",
            }
        except:
            continue


# @pipe.register("agent_trove")
def agent_trove_gen():
    ds = load_dataset("open-thoughts/AgentTrove", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            msgs = repo["conversations"]
            msg = to_chatml(msgs)
            yield {
                "text": msg,
                "category": "agent_trove",
            }
        except:
            continue


langs = "Python, C, Go, Golang"


def stack_v3_gen(supported_langs=None):
    ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, cache_dir=CACHE_DIR)
    if isinstance(supported_langs, str):
        supported_langs = {s.strip().lower() for s in supported_langs.split(",")}
    elif supported_langs is not None:
        supported_langs = {s.strip().lower() for s in supported_langs}
    for repo in ds:
        try:
            for f in repo["files"]:
                if supported_langs is not None and f["language"].lower() not in supported_langs:
                    continue
                category = f["language"]
                if f["language"] not in COMMON_LANG:
                    category = "OtherLanguage"
                yield {
                    "category": category,
                    "text": f["content"],
                }
        except:
            continue


pipe.register("stack_v3", lambda: stack_v3_gen(langs), prefetch=32)


# @pipe.register("reasoning")
def reasoning_gen():
    ds = load_dataset("Qyrou/reasoning-corpus-4K-5M-v1", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            yield {
                "text": repo["ChatML"],
                "category": "reasoning",
            }
        except:
            continue

@pipe.register("claude")
def claude_gen():
    ds = load_dataset("clzoro/Claude-Distills", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            messages = [m for m in (repo.get("messages") or [])
                        if isinstance(m, dict) and m.get("content", "").strip()]
            if not messages:
                continue
            msg = to_chatml(messages)
            yield {
                "text": msg,
                "category": "claude",
            }
        except:
            continue


@pipe.register("rose")
def rose_gen():
    ds = load_dataset("CL-From-Nothing/rose_code_samples", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            prompt = repo.get("prompt") or ""
            response = repo.get("response") or ""
            try:
                rewards = float(repo.get("rewards") or 0)
            except (TypeError, ValueError):
                continue
            if not prompt or not response or rewards <= 0.5:
                continue
            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": "rose",
            }
        except:
            continue

@pipe.register("google_defect")
def google_defect_gen():
    ds = load_dataset("google/code_x_glue_cc_defect_detection", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            func = repo.get("func") or ""
            try:
                target = int(repo.get("target") or 0)
            except (TypeError, ValueError):
                continue
            if not func:
                continue
            prompt = f"Analyze the following code and determine if it contains a defect:\n{func}\n"
            response = "The code contains a defect." if target else "The code does not contain a defect."
            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": "code_defect_detection",
            }
        except:
            continue

@pipe.register("google_function_clone")
def google_function_clone_gen():
    ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        func1 = repo.get("func1") or ""
        func2 = repo.get("func2") or ""
        try:
            label = int(repo.get("label") or 0)
        except (TypeError, ValueError):
            continue
        if not func1 or not func2:
            continue
        if random.random() < 0.5 or label == 0:
            prompt = f"Determine if the following two code snippets are functionally equivalent:\nSnippet 1:\n{func1}\nSnippet 2:\n{func2}\n"
            response = "The two code snippets are functionally equivalent." if label == 1 else "The two code snippets are not functionally equivalent."
        else:
            prompt = f"Write functionally equivalent code for the following snippet:\n{func1}\n"
            response = f"Functionally equivalent code:\n{func2}"
        msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
        yield {
            "text": msg,
            "category": "code_clone_detection",
        }

@pipe.register("code_instruction")
def code_instruction_gen():
    ds = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            instruction = repo["instruction"]
            output = repo["output"]
            msg = to_chatml([{"role": "user", "content": instruction}, {"role": "assistant", "content": output}])
            yield {
                "text": msg,
                "category": "instruction_code",
            }
        except:
            continue


@pipe.register("fineweb", prefetch=8)
def fineweb_gen():
    ds = load_dataset("m-a-p/FineFineWeb", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            if repo["language_score"] < 0.85:
                continue
            yield {
                "text": repo["text"],
                "category": "web",
            }
        except:
            continue


@pipe.register("bigvul")
def bigvul_gen():
    ds = load_dataset("bstee615/bigvul", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            func_after = repo["func_after"]
            func_before = repo["func_before"]
            commit_msg = repo.get("commit_message", "")
            prompt = f"Fix the following vulnerable code:\n{func_before}\n"
            response = f"Fixed code:\n{func_after}"
            if commit_msg:
                response += f"\nExplanation:\n{commit_msg}"

            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": "bigvul",
            }
        except:
            continue


@pipe.register("commitpackft")
def commitpackft_gen(max_chars=10000):
    ds = load_dataset(
        "json",
        data_files="hf://datasets/bigcode/commitpackft/data/*/data.jsonl",
        split="train", streaming=True, cache_dir=CACHE_DIR,
    )
    for repo in ds:
        try:
            old = repo.get("old_contents") or ""
            new = repo.get("new_contents") or ""
            lang = repo.get("lang") or "Unknown"
            subject = (repo.get("subject") or "").strip()
            if not old or not new or old == new:
                continue
            if len(old) + len(new) > max_chars:
                continue
            prompt = f"Update the following {lang} code:\n{old}\n"
            response = f"Updated code:\n{new}"
            if subject:
                response += f"\n\nChange description: {subject}"

            msg = to_chatml([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": f"pack_commit",
            }
        except:
            continue


@pipe.register("x_coder")
def x_coder_gen(max_chars=12000):
    ds = load_dataset("IIGroup/X-Coder-SFT-376k", split="unique_prompt_202k",
                      streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            query = repo.get("query") or ""
            response = repo.get("response") or ""
            if not query or not response:
                continue
            if len(query) + len(response) > max_chars:
                continue
            msg = to_chatml([{"role": "user", "content": query},
                             {"role": "assistant", "content": response}])
            yield {
                "text": msg,
                "category": "x_coder",
            }
        except:
            continue


@pipe.register("code_search_net")
def code_search_net_gen():
    ds = load_dataset("code-search-net/code_search_net", "all", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            func_name = repo.get("func_name") or ""
            language = repo.get("language") or ""
            func_documentation_string = repo.get("func_documentation_string") or ""

            func_code_string = repo.get("func_code_string") or ""

            if not func_name or not language or not func_documentation_string or not func_code_string:
                continue

            prompt = f"Generate a function in {language} that meets the following requirements:\n" \
                     f"Function Name: {func_name}\n" \
                     f"Documentation: {func_documentation_string}\n"

            msg = to_chatml([{"role": "user", "content": prompt},
                             {"role": "assistant", "content": func_code_string}])
            yield {
                "text": msg,
                "category": f"{language}_instruction_code",
            }
        except:
            continue


@pipe.register("code_search_net_raw")
def code_search_net_raw_gen():
    ds = load_dataset("code-search-net/code_search_net", "all", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            whole_func_string = repo.get("whole_func_string") or ""
            language = repo.get("language") or ""
            if not whole_func_string or not language:
                continue
            yield {
                "text": whole_func_string,
                "category": f"{language}"
            }
        except:
            continue


def manusagents_gen():  # DO NOT USE
    ds = load_dataset(
        "Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset",
        split="train", streaming=True, cache_dir=CACHE_DIR
    )
    for repo in ds:
        try:
            raw = repo.get("instruction")
            resp = repo.get("response")
            if not raw or not resp:
                continue

            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    try:
                        parsed = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        continue
            elif isinstance(raw, dict):
                parsed = raw
            else:
                continue

            msgs = parsed.get("messages")
            if not isinstance(msgs, list):
                continue

            messages = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                r = m.get("role") or m.get("from")
                c = m.get("content") or m.get("value")
                if isinstance(r, str) and isinstance(c, str) and r and c:
                    messages.append({"role": r, "content": c})

            if not messages or not isinstance(resp, str) or not resp.strip():
                continue

            messages.append({"role": "assistant", "content": resp})
            yield {"text": to_chatml(messages), "category": "instruction"}
        except:
            continue


def open_math_gen():  # No need for this dataset, we want only coding data
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            problem = repo["problem"]
            solution = repo["generated_solution"]
            _text = f"Problem: {problem}\nSolution: {solution}"
            yield {
                "text": _text,
                "category": "math",
            }
        except:
            continue


def glm_nemotron_codealpaca_gen():  # Too much math
    ds = load_dataset("JessieWei/GLM-5.2-FP8-nemotron-codealpaca-thinking", split="train", streaming=True, cache_dir=CACHE_DIR)
    for repo in ds:
        try:
            conversation = repo["conversations"]
            msg = to_chatml(conversation)
            yield {
                "text": msg,
                "category": "instruction_code_alpaca",
            }
        except:
            continue


if __name__ == "__main__":
    pipe.dry_run = "--dry-run" in sys.argv
    pipe.run(upload_threads=2)
