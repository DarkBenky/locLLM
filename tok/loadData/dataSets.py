import ast
import re
from datasets import load_dataset
from pprint import pprint
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import json
import time
import random

API = "http://91.98.145.193:8823/"
CACHE_DIR = "data/"
CHECKPOINT_FILE = "checkpoint.json"


def send_with_retry(url, payload, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                return res
            if res.status_code >= 500:
                raise RequestException(f"HTTP {res.status_code}")
            return res
        except (ConnectionError, Timeout, RequestException) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"[retry] attempt {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
            time.sleep(delay)
    return None


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
            print(f"[checkpoint] resuming from iter={ckpt['iter']:_}, tokens={ckpt['tokens']:_}")
            return ckpt
    return None

def save_checkpoint(_iter, tokenCount):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"iter": _iter, "tokens": tokenCount}, f)

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


def build_chatml_segments(messages, tools=None):
    segments = []

    def turn(role, body, trainable):
        segments.append((f"<|im_start|>{role}\n", False))
        segments.append((body, trainable))
        segments.append(("<|im_end|>\n", trainable))

    for msg in messages:
        role, trainable = msg["role"], msg.get("trainable", False)

        if role == "system":
            body = msg["content"]
            if tools:
                body += (
                    "\n\nYou have access to the following tools:\n"
                    f"<tools>\n{json.dumps(tools, ensure_ascii=False)}\n</tools>\n\n"
                    'To call a tool, respond with:\n<tool_call>\n'
                    '{"name": "...", "arguments": {...}}\n</tool_call>'
                )
            turn("system", body, False)

        elif role == "user":
            turn("user", msg["content"], False)

        elif role == "assistant":
            if msg.get("tool_calls"):
                calls = "\n".join(
                    "<tool_call>\n"
                    + json.dumps({
                        "name": c["function"]["name"],
                        "arguments": json.loads(c["function"]["arguments"])
                                     if isinstance(c["function"]["arguments"], str)
                                     else c["function"]["arguments"],
                    }, ensure_ascii=False)
                    + "\n</tool_call>"
                    for c in msg["tool_calls"]
                )
                turn("assistant", calls, trainable)
            else:
                body = msg["content"]
                if msg.get("reasoning_content"):
                    body = f"<think>\n{msg['reasoning_content']}\n</think>\n\n{body}"
                turn("assistant", body, trainable)

        elif role == "tool":
            turn("tool", f"<tool_response>\n{msg['content']}\n</tool_response>", False)

        else:
            turn(role, msg["content"], False)

    return segments


def encode_example(segments, sp, ignore_index=-100):
    ids, labels = [], []
    for text, trainable in segments:
        piece_ids = sp.encode(text, out_type=int)
        ids.extend(piece_ids)
        labels.extend(piece_ids if trainable else [ignore_index] * len(piece_ids))
    return ids, labels

def from_chatml(text):
    messages = []
    blocks = text.split("<|im_start|>")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block = block.replace("<|im_end|>", "").strip()
        role, _, content = block.partition("\n")
        messages.append({"role": role.strip(), "content": content.strip()})
    return messages


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


def getNextSample():
    COMMON_LANG = ["Python", "JavaScript", "C++", "Java", "C", "Go", "TypeScript", "Ruby", "Rust", "PHP", "Swift", "C#", "Kotlin", "Scala", "Dart", "Objective-C", "Perl", "Lua", "SQL", "HTML", "CSS", "JSON", "YAML", "Markdown", "XML"]

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

    # No need for this dataset we want only coding staff data
    def open_math_gen():
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


    # DO NOT USE
    def manusagents_gen():
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
    
                parts = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    r = m.get("role") or m.get("from")
                    c = m.get("content") or m.get("value")
                    if isinstance(r, str) and isinstance(c, str) and r and c:
                        parts.append(f"<|im_start|>{r}\n{c}<|im_end|>")
    
                if not parts or not isinstance(resp, str) or not resp.strip():
                    continue
    
                parts.append(f"<|im_start|>assistant\n{resp}<|im_end|>")
                yield {"text": "\n".join(parts), "category": "instruction"}
            except:
                continue

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

    # Too much math
    def glm_nemotron_codealpaca_gen():
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
    
    # gens = [stack_v3_gen(), reasoning_gen(), manusagents_gen(), fineweb_gen(), code_instruction_gen(), open_math_gen(), code_feedback_gen(), nemotron_codealpaca_gen(), code_gen(), tiny_codes_gen(), code_security_gen()]
    # gens = [stack_v3_gen(), code_feedback_gen(), nemotron_codealpaca_gen(), code_gen(), tiny_codes_gen(), code_security_gen()]
    
    langs = "Python, C, Go, Golang"
    
    # gens = [stack_v3_gen(supported_langs=langs)]

    gens = [
        code_alchemy_gen(),
        evo_code_instruction_gen(),
        ultrachat_gen(),
        distillation_qwen_kimi_glm_gen(),
        codealpha_gen(),
        small_talk_gen(),
        code_security_gen(),
        tiny_codes_gen(),
        code_gen(),
        nemotron_codealpaca_gen(),
        code_feedback_gen(),
        star_coder_gen(),
        agent_trove_gen(),
        stack_v3_gen(supported_langs=langs),
        reasoning_gen(),
        code_instruction_gen(),
        fineweb_gen(),
        bigvul_gen(),
        commitpackft_gen(),
    ]

    active = list(range(len(gens)))

    while active:
        for i in active[:]:
            try:
                yield next(gens[i])
            except StopIteration:
                active.remove(i)


if __name__ == "__main__":
    gen = getNextSample()

    ckpt = load_checkpoint()
    start_iter = ckpt["iter"] if ckpt else 0
    tokenCount = ckpt["tokens"] if ckpt else 0
    _iter = 0

    if start_iter > 0:
        print(f"[checkpoint] skipping {start_iter:_} records...")
        for _ in range(start_iter):
            try:
                next(gen)
            except StopIteration:
                break
        print("[checkpoint] skip complete")

    lastUpdate = time.time()
    lastTokenCount = tokenCount

    for rec in gen:
        try:
            res = send_with_retry(API + "/api/receive-data", rec)
        except Exception as e:
            print(f"[fatal] send failed after retries: {e}")
            save_checkpoint(_iter + start_iter, tokenCount)
            print(f"[checkpoint] saved at iter={_iter+start_iter:_}, tokens={tokenCount:_}")
            raise

        if res.status_code != 200:
            continue

        tokenCount += res.json()["token_count"]
        _iter += 1

        if _iter % 16 == 0:
            now = time.time()
            delta_tokens = tokenCount - lastTokenCount
            delta_time = now - lastUpdate
            tokensPerSec = delta_tokens / delta_time if delta_time > 0 else 0
            lastTokenCount = tokenCount
            lastUpdate = now
            print(f"iter {_iter+start_iter:_} tokenCount {tokenCount:_} tokensPerSec {tokensPerSec:_.0f}")

        if _iter % 1000 == 0:
            save_checkpoint(_iter + start_iter, tokenCount)

    save_checkpoint(_iter + start_iter, tokenCount)
    print(f"Done — total iter={_iter+start_iter:_}, tokens={tokenCount:_}")
