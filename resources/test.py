#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple
from urllib import parse, request, error
from urllib.parse import urlsplit

DEFAULT_URL = "https://3z9vr1u26myyp0-10006.proxy.runpod.net/generate_and_validate/"

# A small pool of modern browser UAs to rotate through on retries
BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

def _browserish_headers(url: str, ua_idx: int = 0) -> Dict[str, str]:
    """
    Build a header set that looks like a real browser.
    Avoid Accept-Encoding unless you plan to handle gzip/deflate manually in urllib.
    """
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    return {
        "User-Agent": BROWSER_UAS[ua_idx % len(BROWSER_UAS)],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": origin + "/",
        "Origin": origin,
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded",
    }

def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # strip blanks, preserve full line text
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def sample_prompts(lines: List[str], n: int, seed: int = None) -> List[str]:
    rnd = random.Random(seed)
    if not lines:
        return []
    if n >= len(lines):
        # shuffle all (no replacement)
        lines = lines[:]
        rnd.shuffle(lines)
        return lines
    return rnd.sample(lines, n)

def _reduce_response(prompt: str, body: str) -> Tuple[str, Dict[str, Any]]:
    """
    Try to parse JSON and return the reduced dict; otherwise return an error with the raw body.
    """
    try:
        d = json.loads(body)
        gen = d.get("gen") or {}
        val = d.get("validation") or {}
        out = {
            "prompt": prompt,
            "clip": gen.get("score"),
            "attempts": gen.get("attempts"),
            "elapsed_s": gen.get("elapsed_s"),
            "validation_score": val.get("score"),
            "iqa": val.get("iqa"),
            "alignment_score": val.get("alignment_score"),
            "ssim": val.get("ssim"),
            "lpips": val.get("lpips"),
        }
        return prompt, out
    except Exception:
        return prompt, {"prompt": prompt, "error": True, "response_raw": body}

def post_prompt(url: str, prompt: str, timeout: float = 120.0, ua_idx: int = 0) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (prompt, reduced_json_dict). On error, returns an entry with error=True and response_raw.
    Sends browser-like headers to avoid Cloudflare 1010 (bot) blocks.
    """
    payload = parse.urlencode({"prompt": prompt}).encode("utf-8")
    req = request.Request(url, data=payload, method="POST")
    for k, v in _browserish_headers(url, ua_idx).items():
        req.add_header(k, v)

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            # If CF lets us through, response should be JSON from your service
            body = resp.read().decode("utf-8", errors="replace")
            return _reduce_response(prompt, body)
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return prompt, {"prompt": prompt, "error": True, "status": getattr(e, "code", None), "response_raw": raw}
    except Exception as e:
        return prompt, {"prompt": prompt, "error": True, "exception": repr(e)}

def post_with_retries(url: str, prompt: str, timeout: float, retries: int, backoff_s: float) -> Tuple[str, Dict[str, Any]]:
    """
    Retries with exponential backoff and rotates UA each attempt to dodge CF bot heuristics.
    Adds a small jitter if a 403 occurs.
    """
    for attempt in range(retries + 1):
        pr, out = post_prompt(url, prompt, timeout=timeout, ua_idx=attempt)
        if not out.get("error"):
            return pr, out

        # If explicitly blocked (403 / Cloudflare), quick jitter then continue
        if out.get("status") == 403 and attempt < retries:
            time.sleep(0.25 + random.random() * 0.25)

        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt))

    return pr, out  # last error

def main():
    ap = argparse.ArgumentParser(description="Run N random prompts against an endpoint and collect scores.")
    ap.add_argument("n", type=int, help="Number of prompts to run")
    ap.add_argument("--prompts", default="prompts.txt", help="Path to prompts file (default: prompts.txt)")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--out", default=None, help="Output filename (.json). Default: results_YYYYMMDD_HHMMSS.json")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per call in seconds (default: 120)")
    ap.add_argument("--concurrency", type=int, default=1, help="Number of parallel workers (default: 1)")
    ap.add_argument("--retries", type=int, default=2, help="Retries per prompt on failure (default: 2)")
    ap.add_argument("--backoff", type=float, default=0.75, help="Initial backoff seconds between retries (default: 0.75)")
    args = ap.parse_args()

    if args.n <= 0:
        print("n must be > 0", file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(args.prompts):
        print(f"Prompts file not found: {args.prompts}", file=sys.stderr)
        sys.exit(2)

    lines = read_prompts(args.prompts)
    sampled = sample_prompts(lines, args.n, seed=args.seed)

    if not sampled:
        print("No non-empty prompts found to run.", file=sys.stderr)
        sys.exit(2)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"results_{stamp}.json"

    print(f"Selected {len(sampled)} prompts (from {len(lines)} lines).")
    print(f"Posting to: {args.url}")
    print(f"Concurrency: {args.concurrency} | Retries: {args.retries} | Timeout: {args.timeout}s")
    print("Running...")

    results: List[Dict[str, Any]] = []

    if args.concurrency <= 1:
        # Sequential
        for i, p in enumerate(sampled, 1):
            print(f"[{i}/{len(sampled)}] {p}")
            _, out = post_with_retries(args.url, p, args.timeout, args.retries, args.backoff)
            results.append(out)
    else:
        # Parallel
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(post_with_retries, args.url, p, args.timeout, args.retries, args.backoff): p
                for p in sampled
            }
            done = 0
            for fut in cf.as_completed(futs):
                done += 1
                p = futs[fut]
                try:
                    _, out = fut.result()
                except Exception as e:
                    out = {"prompt": p, "error": True, "exception": repr(e)}
                print(f"[{done}/{len(sampled)}] {p}")
                results.append(out)

    # Write output JSON array
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Simple summary
    ok = sum(1 for r in results if not r.get("error"))
    errs = len(results) - ok
    print(f"Done. Wrote {len(results)} entries to: {out_path}")
    print(f"Success: {ok} | Errors: {errs}")

if __name__ == "__main__":
    main()
