"""Benchmarks for external capability exposure.

    tokens      the model-visible cost of a family of 5, 10, 25 and 50
                capabilities: every schema up front, against the index, and
                the index after one capability has been described
    discovery   whether a model picks the right capability from an index of
                10, 25 and 50 (catalog.CASES), asked which one it would
                describe first
    latency     connecting to, listing, describing and invoking a provider
                (the synthetic MCP server of the tests), cold and cached

Usage:
    bench.py tokens [--tokenize URL]
    bench.py discovery --model URL [--model-name NAME]
    bench.py latency
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from harness import capabilities as cap  # noqa: E402
import catalog                                  # noqa: E402

PROVIDER = cap.ProviderConfig(id="ops", transport="stdio", command=("none",),
                              read=frozenset(tool["name"] for tool in catalog.TOOLS))
SIZES = (5, 10, 25, 50)


def family(size):
    return [cap.capability(PROVIDER, tool["name"], tool["description"], tool["inputSchema"],
                           provenance="catalog") for tool in catalog.TOOLS[:size]]


def counter(url):
    if not url:
        return cap.estimate

    def tokens(text):
        request = urllib.request.Request(url.rstrip("/") + "/tokenize",
                                         data=json.dumps({"content": text}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return len(json.load(response)["tokens"])

    return tokens


def tokens_bench(url):
    count = counter(url)
    print(f"tokens ({'model tokenizer' if url else 'estimate, chars/3'}):")
    print("  size  policy-mode  full-schemas  index  index+1-described  saving-initial")

    for size in SIZES:
        items = family(size)
        full = count(cap.render(items, cap.DIRECT))
        index = count(cap.render(items, cap.INDEXED))
        one = index + count(cap.described(items[0]))
        print(f"  {size:>4}  {cap.exposure(items):>11}  {full:>12}  {index:>5}  {one:>17}  "
              f"{100 * (1 - index / full):>13.0f}%")


def ask(url, name, system, question):
    body = {"model": name, "temperature": 0, "max_tokens": 40,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": question}]}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def discovery_bench(url, name):
    print("discovery (which capability would be described first):")

    for size in (10, 25, 50):
        section = cap.render(family(size), cap.INDEXED)
        system = ("You are a coding agent working in a software project." + section)
        right, mistakes = 0, []

        for question, target in catalog.CASES:
            answer = ask(url, name, system, question + "\n\nWhich capability would you "
                         "describe first to do this? Reply with its id only.")
            # Scored as the gateway would resolve it: a full id, or a bare
            # name that exactly one capability carries.
            chosen = (re.findall(r"[\w.-]+(?:/[\w.-]+)?", answer) or [""])[0]
            chosen = chosen if "/" in chosen else f"ops/{chosen}"

            if chosen == f"ops/{target}":
                right += 1
            else:
                mistakes.append(f"{question[:60]!r}: chose {chosen}, expected ops/{target}")

        print(f"  {size:>3} capabilities: {right}/{len(catalog.CASES)} correct")

        for mistake in mistakes:
            print(f"      {mistake}")


def latency_bench():
    from harness import capability_gateway as gw
    from harness import mcp_provider

    server = str(HERE.parent.parent / "tests" / "mcp_fixture_server.py")
    rows = {key: [] for key in ("connect+list", "cached list", "describe", "invoke")}

    for _ in range(5):
        config = cap.ProviderConfig(id="fixture", transport="stdio",
                                    command=(sys.executable, server, "--extra", "47"),
                                    scope=("generic",), read=frozenset({"lookup_component"}))
        registry = gw.Registry([config], factory=mcp_provider.provider_for)
        door = gw.Gateway(registry, ("fixture",))

        clock = time.perf_counter()
        door.prepare()
        rows["connect+list"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        registry.provider("fixture").list()
        rows["cached list"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        door.run("spear-capability describe fixture/lookup_component")
        rows["describe"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        door.run("spear-capability invoke fixture/lookup_component '{\"name\": \"uart\"}'")
        rows["invoke"].append(time.perf_counter() - clock)
        registry.close()

    print("latency (synthetic MCP server, 50 tools, 5 runs; ms median / max):")

    for key, values in rows.items():
        print(f"  {key:13} {1000 * statistics.median(values):8.2f} / "
              f"{1000 * max(values):8.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("tokens", "discovery", "latency"))
    parser.add_argument("--tokenize", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default="qwen3")
    args = parser.parse_args(argv)

    if args.what == "tokens":
        tokens_bench(args.tokenize)
    elif args.what == "discovery":
        discovery_bench(args.model, args.model_name)
    else:
        latency_bench()


if __name__ == "__main__":
    main()
