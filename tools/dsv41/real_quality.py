#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash real-weights checks against a running vllm (dsv41 real).

  real_quality.py checks --tag run1 [q1a q1b q2 q3 q4 q5]
      Each check saves {request, response, wall_s} as <outdir>/<tag>-<check>.json
      and prints a trimmed summary. q1a/q1b: chat mode (reasoning_effort none);
      q2: thinking mode, expects reasoning_content; q3: get_weather tool call and
      the tool-result turn; q4: ~20k-token needle at temperature 0; q5: 3-turn
      conversation with tools defined (drop_thinking rules).
  real_quality.py compare run1 run2 q1a q2
      Diff the saved greedy outputs of two tags (first differing char).
  real_quality.py det --tag run2 [--n 5] [--max-tokens 200]
      Send one greedy request N times with top logprobs; report distinct
      completions and the logprob gap at the first divergent token.

Defaults: --base http://127.0.0.1:8080, --outdir ~/dsv41-test/real.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request

CAPITAL_PROMPT = (
    "What is the capital of Australia and what year did it become the capital?"
)
MERGE_PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted"
    " list, with a docstring."
)
PACK_PROMPT = (
    "Given that, should I still go, and what should I pack? One short paragraph."
)
NEEDLE_FACT = (
    "Important note for later: the access code for the Kestrel storage "
    "locker is 7391-MARLIN-42. "
)
NEEDLE_QUESTION = (
    "  Question: what is the access code for the Kestrel storage locker? "
    "Answer with just the code."
)
SHEEP_PROMPT = (
    "A farmer has 17 sheep. All but 9 run away, and then she buys twice as "
    "many as she has left. How many sheep does she have now? Give the final "
    "number."
)
PICNIC_PROMPT = (
    "I am planning a picnic this weekend. Which of Sydney, Melbourne or Hobart "
    "usually has the mildest spring weather? Keep it brief."
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Sydney"},
            },
            "required": ["city"],
        },
    },
}


def post(base, body, timeout=1800):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        resp = {"http_error": e.code, "body": e.read().decode(errors="replace")}
    return resp, time.time() - t0


def trim(s, n=600):
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... [{len(s)} chars]"


def summarize(name, resp, wall):
    if "http_error" in resp:
        print(f"[{name}] HTTP {resp['http_error']}: {trim(resp['body'])}")
        return
    ch = resp["choices"][0]
    m = ch["message"]
    u = resp.get("usage", {})
    ct = u.get("completion_tokens", 0)
    print(
        f"[{name}] finish={ch.get('finish_reason')} "
        f"prompt_tokens={u.get('prompt_tokens')} completion_tokens={ct} "
        f"wall={wall:.1f}s tok/s={ct / wall if wall else 0:.1f}"
    )
    rc = m.get("reasoning_content") or m.get("reasoning")
    if rc:
        print(f"  reasoning_content ({len(rc)} chars): {trim(rc, 400)!r}")
    print(f"  content: {trim(m.get('content'))!r}")
    if m.get("tool_calls"):
        print(f"  tool_calls: {json.dumps(m['tool_calls'])}")


def run(name, base, outdir, tag, body, timeout=1800):
    resp, wall = post(base, body, timeout)
    path = os.path.join(outdir, f"{tag}-{name}.json")
    with open(path, "w") as f:
        json.dump({"request": body, "response": resp, "wall_s": wall}, f, indent=1)
    summarize(name, resp, wall)
    return resp


def q1a(base, outdir, tag, temp):
    return run(
        "q1a",
        base,
        outdir,
        tag,
        {
            "model": "dsv41",
            "temperature": temp,
            "max_tokens": 300,
            "reasoning_effort": "none",
            "messages": [
                {
                    "role": "user",
                    "content": CAPITAL_PROMPT,
                }
            ],
        },
    )


def q1b(base, outdir, tag, temp):
    return run(
        "q1b",
        base,
        outdir,
        tag,
        {
            "model": "dsv41",
            "temperature": temp,
            "max_tokens": 500,
            "reasoning_effort": "none",
            "messages": [
                {
                    "role": "user",
                    "content": MERGE_PROMPT,
                }
            ],
        },
    )


def q2(base, outdir, tag, temp):
    return run(
        "q2",
        base,
        outdir,
        tag,
        {
            "model": "dsv41",
            "temperature": temp,
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": SHEEP_PROMPT,
                }
            ],
        },
    )


def q3(base, outdir, tag, temp):
    messages = [{"role": "user", "content": "What is the weather in Sydney right now?"}]
    r1 = run(
        "q3a",
        base,
        outdir,
        tag,
        {
            "model": "dsv41",
            "temperature": temp,
            "max_tokens": 1500,
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "reasoning_effort": "none",
            "messages": messages,
        },
    )
    if "http_error" in r1:
        return r1
    m1 = message(r1)
    tcs = m1.get("tool_calls") or []
    if not tcs:
        print("  q3: NO tool_calls returned; skipping tool-result turn")
        return r1
    assistant = {
        "role": "assistant",
        "content": m1.get("content") or "",
        "tool_calls": tcs,
    }
    if m1.get("reasoning_content"):
        assistant["reasoning_content"] = m1["reasoning_content"]
    messages = messages + [
        assistant,
        {
            "role": "tool",
            "tool_call_id": tcs[0]["id"],
            "content": json.dumps(
                {
                    "city": "Sydney",
                    "temperature_c": 21,
                    "condition": "partly cloudy",
                    "wind_kph": 18,
                }
            ),
        },
    ]
    return run(
        "q3b",
        base,
        outdir,
        tag,
        {
            "model": "dsv41",
            "temperature": temp,
            "max_tokens": 600,
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "reasoning_effort": "none",
            "messages": messages,
        },
    )


FILLER = [
    (
        "The harbour ferries run every twenty minutes on weekdays and every "
        "half hour at weekends."
    ),
    (
        "Granite weathers slowly, shedding grains of quartz and feldspar into "
        "the creek beds below."
    ),
    (
        "The bakery on the corner sells rye loaves on Tuesdays and sourdough on"
        " every other day."
    ),
    (
        "A pendulum clock loses time when the room warms, because the brass rod"
        " lengthens a fraction."
    ),
    (
        "The council repainted the pedestrian crossings after the winter storms"
        " faded the stripes."
    ),
    (
        "Migrating shorebirds rest on the mudflats before crossing the strait "
        "toward the southern islands."
    ),
    (
        "The library extended its opening hours during exam season, closing at "
        "eleven instead of nine."
    ),
    (
        "Old copper pipes develop a green patina where the joints were soldered"
        " decades ago."
    ),
    (
        "The orchard's late apples are pressed into juice that is bottled "
        "without added sugar."
    ),
    (
        "Wind turbines on the ridge turn slowly on calm mornings and feather "
        "their blades in gales."
    ),
    (
        "The tram depot keeps two heritage cars that run on public holidays for"
        " enthusiasts."
    ),
    (
        "Basalt columns along the coast formed as a lava flow cooled and "
        "contracted into hexagons."
    ),
]


def needle_prompt(target_tokens=20000, seed=7):
    rng = random.Random(seed)
    # roughly 4 chars per token; each filler sentence is ~20 tokens
    n_sent = target_tokens // 18
    fact = NEEDLE_FACT
    sents = [rng.choice(FILLER) for _ in range(n_sent)]
    sents.insert(len(sents) // 2, fact)
    body = " ".join(sents)
    return "Read the following notes carefully.\n\n" + body + NEEDLE_QUESTION


def q4(base, outdir, tag, temp):
    return run(
        "q4",
        base,
        outdir,
        tag,
        {
            "model": "dsv41",
            "temperature": 0,
            "max_tokens": 100,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": needle_prompt()}],
        },
        timeout=3600,
    )


def q5(base, outdir, tag, temp):
    common = {
        "model": "dsv41",
        "temperature": temp,
        "max_tokens": 1500,
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
    }
    messages = [
        {
            "role": "user",
            "content": PICNIC_PROMPT,
        }
    ]
    r1 = run("q5a", base, outdir, tag, {**common, "messages": messages})
    if "http_error" in r1:
        return r1
    m1 = message(r1)
    a1 = {"role": "assistant", "content": m1.get("content") or ""}
    if m1.get("reasoning_content"):
        a1["reasoning_content"] = m1["reasoning_content"]
    if m1.get("tool_calls"):
        a1["tool_calls"] = m1["tool_calls"]
    messages = messages + [
        a1,
        {
            "role": "user",
            "content": "Thanks. Now check the actual weather in Hobart for me.",
        },
    ]
    r2 = run("q5b", base, outdir, tag, {**common, "messages": messages})
    if "http_error" in r2:
        return r2
    m2 = message(r2)
    a2 = {"role": "assistant", "content": m2.get("content") or ""}
    if m2.get("reasoning_content"):
        a2["reasoning_content"] = m2["reasoning_content"]
    tail = []
    if m2.get("tool_calls"):
        a2["tool_calls"] = m2["tool_calls"]
        tail = [
            {
                "role": "tool",
                "tool_call_id": m2["tool_calls"][0]["id"],
                "content": json.dumps(
                    {
                        "city": "Hobart",
                        "temperature_c": 14,
                        "condition": "showers",
                        "wind_kph": 30,
                    }
                ),
            }
        ]
    else:
        print("  q5b: no tool call; continuing without a tool result")
    messages = (
        messages
        + [a2]
        + tail
        + [
            {
                "role": "user",
                "content": PACK_PROMPT,
            }
        ]
    )
    return run("q5c", base, outdir, tag, {**common, "messages": messages})


CHECKS = {"q1a": q1a, "q1b": q1b, "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def message(resp):
    return resp["choices"][0]["message"]


def cmd_checks(a):
    os.makedirs(a.outdir, exist_ok=True)
    for c in a.checks or list(CHECKS):
        if c not in CHECKS:
            sys.exit(f"unknown check {c}")
        CHECKS[c](a.base, a.outdir, a.tag, a.temp)
        sys.stdout.flush()


def load_message(outdir, tag, check):
    with open(os.path.join(outdir, f"{tag}-{check}.json")) as fh:
        return message(json.load(fh)["response"])


def cmd_compare(a):
    for c in a.checks:
        ra, rb = (load_message(a.outdir, t, c) for t in (a.tag_a, a.tag_b))
        for f in ("reasoning_content", "reasoning", "content"):
            x, y = ra.get(f) or "", rb.get(f) or ""
            if not x and not y:
                continue
            if x == y:
                print(f"{c}.{f}: IDENTICAL ({len(x)} chars)")
                continue
            i = next(
                (i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))
            )
            print(
                f"{c}.{f}: DIFFER at char {i} of {len(x)}/{len(y)}: "
                f"{a.tag_a}={x[i : i + 80]!r} {a.tag_b}={y[i : i + 80]!r}"
            )


def cmd_det(a):
    body = {
        "model": "dsv41",
        "temperature": 0,
        "max_tokens": a.max_tokens,
        "reasoning_effort": "none",
        "logprobs": True,
        "top_logprobs": 3,
        "seed": 0,
        "messages": [
            {
                "role": "user",
                "content": CAPITAL_PROMPT,
            }
        ],
    }
    runs = []
    for i in range(a.n):
        resp, wall = post(a.base, body)
        lp = resp["choices"][0]["logprobs"]["content"]
        runs.append(([x["token"] for x in lp], lp))
        ct = resp["usage"]["completion_tokens"]
        print(f"run {i}: {ct} tok, wall {wall:.2f}s, {ct / wall:.1f} tok/s")
        with open(os.path.join(a.outdir, f"{a.tag}-det{i}.json"), "w") as f:
            json.dump({"request": body, "response": resp, "wall_s": wall}, f, indent=1)
    ref_toks, ref_lp = runs[0]
    print(f"distinct completions: {len({tuple(t) for t, _ in runs})}/{a.n}")
    for i, (toks, lp) in enumerate(runs[1:], 1):
        k = next((j for j, (p, q) in enumerate(zip(ref_toks, toks)) if p != q), None)
        if k is None:
            print(
                f"run {i}: identical to run 0 for "
                f"{min(len(toks), len(ref_toks))} tokens"
            )
            continue
        fa = {x["token"]: round(x["logprob"], 4) for x in ref_lp[k]["top_logprobs"]}
        fb = {x["token"]: round(x["logprob"], 4) for x in lp[k]["top_logprobs"]}
        print(
            f"run {i}: first divergence at token {k}: run0={ref_toks[k]!r} top={fa} | "
            f"run{i}={toks[k]!r} top={fb}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--outdir", default=os.path.expanduser("~/dsv41-test/real"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("checks")
    p.add_argument("--tag", required=True)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("checks", nargs="*")
    p.set_defaults(fn=cmd_checks)
    p = sub.add_parser("compare")
    p.add_argument("tag_a")
    p.add_argument("tag_b")
    p.add_argument("checks", nargs="+")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("det")
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=200)
    p.set_defaults(fn=cmd_det)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
