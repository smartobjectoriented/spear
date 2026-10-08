"""Benchmarks for workspace knowledge exposure.

    tokens     what a turn is shown of 5, 10, 25, 50 and 100 records: every
               record in full, against the selection (whole when small, an
               index when not), and the cost of one record shown on demand
    discovery  whether a model picks the record a request needs from an index
               of 10, 25, 50 and 100; the requests avoid the records' subjects,
               so the index has to be read rather than matched
    latency    opening the store, querying a workspace, checking its sources,
               rendering, showing one record and writing one

Usage:
    bench.py tokens [--tokenize URL]
    bench.py discovery --model URL [--model-name NAME]
    bench.py latency

The records are invented; they describe no real project.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from context import workspace_knowledge as wk  # noqa: E402

FACTS = [
    (wk.Kind.PROJECT_FACT, "thermal driver", "The thermal sensor driver lives in drivers/hwmon/tmp_sense.c."),
    (wk.Kind.BUILD_FACT, "boot image", "The boot image is assembled by the image-boot recipe in build/meta-boot."),
    (wk.Kind.ARCHITECTURE_FACT, "rt link", "The Linux side talks to the real-time core through RPMsg channels."),
    (wk.Kind.COMPONENT_RELATION, "display stack", "The display stack depends on the lvgl port in libs/gfx."),
    (wk.Kind.DECISION, "logging backend", "The team replaced the ring-buffer logger with the trace framework in 2025."),
    (wk.Kind.COMMAND_KNOWLEDGE, "parser regression", "make test-parser reproduces the parser regression in under a minute."),
    (wk.Kind.PROJECT_FACT, "firmware entry", "The firmware entry point is src/main.c, function board_main."),
    (wk.Kind.BUILD_FACT, "flash layout", "The flash partition layout is defined in board/partitions.csv."),
    (wk.Kind.ARCHITECTURE_FACT, "update scheme", "Firmware updates use an A/B slot scheme with a rollback counter."),
    (wk.Kind.COMPONENT_RELATION, "modem service", "The modem service needs the power manager to be running first."),
]

CASES = [
    ("Where is the code that reads the temperature sensor?", 0),
    ("Which recipe builds the image the board boots from?", 1),
    ("How does Linux exchange messages with the realtime cores?", 2),
    ("What graphics library does the screen code rely on?", 3),
    ("Which tracing mechanism should new log statements go through?", 4),
    ("What is the quickest way to reproduce the parsing bug?", 5),
    ("Where does execution start when the device powers on?", 6),
    ("Where are the sizes of the storage partitions declared?", 7),
    ("How does an update recover if the new version fails to start?", 8),
    ("What has to be started before the cellular service?", 9),
]


def records(store, workspace, size):
    found = []

    for index in range(size):
        if index < len(FACTS):
            kind, subject, statement = FACTS[index]
        else:
            kind, subject = wk.Kind.PROJECT_FACT, f"module m{index:03d}"
            statement = (f"Module m{index:03d} is maintained by team {index % 7} and lives in "
                         f"src/modules/m{index:03d}/.")

        found.append(store.add(workspace, kind=kind, subject=subject, statement=statement,
                               provenance=wk.Provenance.USER_CONFIRMED))

    return found


def counter(url):
    if not url:
        return wk.estimate

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
    print("  records  mode     all-in-full  selection  +1-shown  saving")

    for size in (5, 10, 25, 50, 100):
        store = wk.KnowledgeStore(":memory:")
        listed = records(store, "bench", size)
        everything = count(wk.HEADER + "".join(wk.full(record) + "\n" for record in listed))
        selection = wk.select(listed, "")
        shown = count(selection.text)
        one = shown + count(wk.detail(listed[0], store.history("bench", listed[0].record_id)))
        print(f"  {size:>7}  {selection.mode:7}  {everything:>11}  {shown:>9}  {one:>8}  "
              f"{100 * (1 - shown / everything):>5.0f}%")
        store.close()


def ask(url, name, system, question):
    body = {"model": name, "temperature": 0, "max_tokens": 30,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": question}]}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def discovery_bench(url, name, chars):
    print(f"discovery (which record would be shown in full first; index lines carry "
          f"{chars} characters of the statement):")

    for size in (10, 25, 50, 100):
        store = wk.KnowledgeStore(":memory:")
        listed = records(store, "bench", size)
        # Forced to the index, whatever the size: what is measured is
        # whether the index alone leads to the right record.
        text = wk.HEADER + "Records, by id:\n" + wk.index(listed, chars)
        right, mistakes = 0, []

        for question, target in CASES:
            answer = ask(url, name, "You are a coding agent working in a software project."
                         + text, question + "\n\nWhich record id would you read in full to "
                         "answer this? Reply with the id only.")
            chosen = (re.findall(r"k-[0-9a-f]{12}", answer) or [answer.strip()[:20]])[0]

            if chosen == listed[target].record_id:
                right += 1
            else:
                picked = next((r.subject for r in listed if r.record_id == chosen), chosen)
                mistakes.append(f"{question[:55]!r}: chose {picked!r}, expected "
                                f"{listed[target].subject!r}")

        print(f"  {size:>3} records: {right}/{len(CASES)} correct")

        for mistake in mistakes:
            print(f"      {mistake}")

        store.close()


def latency_bench():
    rows = {key: [] for key in ("open", "query", "stale check", "render", "detail", "write")}

    for _ in range(5):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "knowledge.sqlite3")
        Path(directory, "Makefile").write_text("all: firmware.elf\n")
        store = wk.KnowledgeStore(path)
        listed = records(store, "bench", 100)
        source, _ = wk.source_evidence(directory, "Makefile", "all: firmware.elf")

        for index in range(20):
            store.add("bench", kind=wk.Kind.BUILD_FACT, subject=f"target {index}",
                      statement=f"Target {index} builds firmware.elf.",
                      provenance=wk.Provenance.PROJECT_SOURCE, sources=(source,))

        store.close()

        for key, action in (("open", lambda: wk.KnowledgeStore(path)),):
            clock = time.perf_counter()
            store = action()
            rows[key].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        active = store.list("bench", (wk.Lifecycle.ACTIVE,))
        rows["query"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        store.validate("bench", directory)
        rows["stale check"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        wk.select(active, "who owns module m042?")
        rows["render"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        wk.detail(store.get("bench", listed[3].record_id),
                  store.history("bench", listed[3].record_id))
        rows["detail"].append(time.perf_counter() - clock)

        clock = time.perf_counter()
        store.add("bench", kind=wk.Kind.DECISION, subject="bench", statement="A decision.",
                  provenance=wk.Provenance.USER_CONFIRMED)
        rows["write"].append(time.perf_counter() - clock)
        store.close()

    print("latency (120 records, 20 bound to a file; 5 runs; ms median / max):")

    for key, values in rows.items():
        print(f"  {key:12} {1000 * statistics.median(values):8.2f} / {1000 * max(values):8.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("tokens", "discovery", "latency"))
    parser.add_argument("--tokenize", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-name", default="qwen3")
    parser.add_argument("--chars", type=int, default=wk.SUMMARY_CHARS)
    args = parser.parse_args(argv)

    if args.what == "tokens":
        tokens_bench(args.tokenize)
    elif args.what == "discovery":
        discovery_bench(args.model, args.model_name, args.chars)
    else:
        latency_bench()


if __name__ == "__main__":
    main()
