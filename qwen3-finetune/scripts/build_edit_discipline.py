#!/usr/bin/env python3
"""Build the EDIT-DISCIPLINE QLoRA dataset for the dense Coder-32B.

Goal: teach the model to answer "improve <file>" with TARGETED edit_file calls
that change only what needs changing and PRESERVE everything else — every
declaration, every CLI option (incl. positional args), and a third-party
copyright header. This directly fixes the observed failure where a full
rewrite silently drops functionality and stops compiling.

Correct-by-construction: each example anchors on the REAL file. We author edits
as (space-typed anchor, replacement); the builder normalizes whitespace to
LOCATE the real region, extracts the file's ACTUAL bytes (tabs and all) as
old_text, and re-indents new_text to the file's style. It ERRORS if an anchor
does not uniquely match — so a mis-typed anchor can never poison the dataset.

Output: data/edit_discipline_train.jsonl + _val.jsonl in {"messages": [...]}.
"""
import json
import os
import re
import sys
from pathlib import Path

SRC = Path(os.environ.get("SO3_SRC",
                          os.path.expanduser("~/soo/so3/so3/usr/src")))
OUT = Path(__file__).resolve().parent.parent / "data"

SYSTEM = (
    "Tu es SPEAR, l'assistant de code pour SO3 (apps dans so3/usr/src). "
    "Pour AMÉLIORER un fichier, n'émets PAS une réécriture complète : produis "
    "un ou plusieurs appels edit_file qui ne touchent QUE les lignes à changer. "
    "Conserve toujours chaque déclaration globale, chaque fonction, chaque "
    "option de ligne de commande (y compris l'argument positionnel) et "
    "l'en-tête de copyright existant. Indentation : tabulations (style noyau)."
)


def canon(s):
    return re.sub(r"[ \t]+", " ", s.strip())


def detect_space_unit(lines):
    units = [len(l) - len(l.lstrip(" ")) for l in lines
             if l[:1] == " " and l.strip()]
    units = [u for u in units if u]
    return min(units) if units else 4


def reindent_tabs(text, unit):
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        nlead = len(line) - len(stripped)
        if nlead and "\t" not in line[:nlead]:
            out.append("\t" * (nlead // unit) + " " * (nlead % unit) + stripped)
        else:
            out.append(line)
    return "\n".join(out)


def locate(content, anchor):
    """Return the file's REAL substring matching `anchor` (whitespace-tolerant),
    plus a tab-reindented note flag. Raises if not a unique match."""
    flines = content.split("\n")
    olines = anchor.rstrip("\n").split("\n")
    n = len(olines)
    oc = [canon(l) for l in olines]
    hits = [i for i in range(len(flines) - n + 1)
            if [canon(flines[i + k]) for k in range(n)] == oc]
    if len(hits) != 1:
        raise SystemExit(f"ANCHOR not unique ({len(hits)} hits):\n{anchor!r}")
    i = hits[0]
    real = "\n".join(flines[i:i + n])
    return real


def edit_xml(path, old_text, new_text):
    return ("<tool_call>\n<function=edit_file>\n"
            f"<parameter=path>{path}</parameter>\n"
            f"<parameter=old_text>{old_text}</parameter>\n"
            f"<parameter=new_text>{new_text}</parameter>\n"
            "</function>\n</tool_call>")


def make_example(relpath, intro, edits, instruction=None):
    """edits: list of (anchor_spaces, replacement_spaces). Builds a chat
    example whose assistant turn issues verified edit_file calls."""
    fpath = SRC / Path(relpath).name
    content = fpath.read_text()
    blocks, summary = [], []
    for anchor, repl in edits:
        real_old = locate(content, anchor)
        unit = detect_space_unit(anchor.rstrip("\n").split("\n")) or 4
        # re-indent replacement to the file's tab style if the region is tabbed
        region_tabbed = "\t" in real_old[:len(real_old) - len(real_old.lstrip())]
        new_text = reindent_tabs(repl, unit) if region_tabbed else repl
        if content.count(real_old) != 1:
            raise SystemExit(f"old not unique in {relpath}: {real_old!r}")
        blocks.append(edit_xml(relpath, real_old, new_text))
    user = (instruction or f"Améliore {relpath}.") + \
        f"\n\nContenu actuel :\n```c\n{content}```"
    assistant = intro + "\n\n" + "\n".join(blocks)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


# ── authored, correct, targeted improvements (anchors typed with spaces;
#    the builder maps them to the file's real tab bytes) ──────────────────
EXAMPLES = []

# ping.c — THE failure family: improve error handling WITHOUT dropping the
# positional destination parsing or any option. Targeted edits only.
EXAMPLES.append(make_example(
    "so3/usr/src/ping.c",
    "Je renforce la gestion d'erreur du socket sans toucher au reste "
    "(parsing des arguments, destination positionnelle et en-tête conservés).",
    [(
        "    if (s < 0) {\n"
        "        printf(\"Impossible to obtain a socket file descriptor!!\\n\");\n"
        "        return 1;\n"
        "    }",
        "    if (s < 0) {\n"
        "        perror(\"ping: socket\");\n"
        "        return 1;\n"
        "    }",
    )],
    instruction="Améliore la gestion d'erreur du socket dans so3/usr/src/ping.c.",
))

# ping.c — replace the magic ICMP reply type 69 with a readable comparison,
# nothing else changed.
EXAMPLES.append(make_example(
    "so3/usr/src/ping.c",
    "Je remplace le nombre magique 69 (type ICMP echo reply) par une comparaison "
    "lisible ; aucune autre ligne n'est modifiée.",
    [(
        "        if (!(packet.hdr.type == 69 && packet.hdr.code == 0)) {",
        "        if (!(packet.hdr.type == ICMP_ECHOREPLY && packet.hdr.code == 0)) {",
    )],
    instruction="Dans so3/usr/src/ping.c, remplace le type ICMP magique 69 par un nom lisible.",
))

# echo.c — fix the trailing space (print a separator BEFORE args 2..n) without
# rewriting the file or touching the header.
EXAMPLES.append(make_example(
    "so3/usr/src/echo.c",
    "Je corrige l'espace en trop en fin de ligne en imprimant le séparateur "
    "avant chaque argument sauf le premier. Le reste est inchangé.",
    [(
        "    for (i = 1; i < argc; i++)\n"
        "        printf(\"%s \", argv[i]);",
        "    for (i = 1; i < argc; i++) {\n"
        "        if (i > 1)\n"
        "            printf(\" \");\n"
        "        printf(\"%s\", argv[i]);\n"
        "    }",
    )],
    instruction="Corrige l'espace superflu en fin de ligne dans so3/usr/src/echo.c.",
))

# time.c — the loop spins as fast as possible; add a 1s pause. Needs unistd.h.
# Two targeted edits: add the include, add the sleep. Nothing else touched.
EXAMPLES.append(make_example(
    "so3/usr/src/time.c",
    "J'ajoute une pause d'une seconde entre deux affichages (la boucle "
    "saturait le CPU). Deux éditions ciblées : l'include et l'appel sleep.",
    [
        ("#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>",
         "#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n#include <unistd.h>"),
        ("        printf(\"# time(s) : %\" PRIu64 \"  time(us) : %\" PRIu64 \"\\n\", tv.tv_sec, tv.tv_usec);",
         "        printf(\"# time(s) : %\" PRIu64 \"  time(us) : %\" PRIu64 \"\\n\", tv.tv_sec, tv.tv_usec);\n\n        sleep(1);"),
    ],
    instruction="Évite que so3/usr/src/time.c sature le CPU (boucle trop serrée).",
))

# cat.c — already clean; teach RESTRAINT: report the errno reason on open
# failure, one targeted edit, header and structure preserved.
EXAMPLES.append(make_example(
    "so3/usr/src/cat.c",
    "J'améliore le message d'erreur d'ouverture en utilisant perror (raison "
    "réelle) ; une seule édition ciblée, tout le reste est conservé.",
    [(
        "        if (fd < 0) {\n"
        "            printf(\"cat: %s: cannot open\\n\", argv[i]);\n"
        "            rc = 1;\n"
        "            continue;\n"
        "        }",
        "        if (fd < 0) {\n"
        "            perror(argv[i]);\n"
        "            rc = 1;\n"
        "            continue;\n"
        "        }",
    )],
    instruction="Améliore le message d'erreur d'ouverture dans so3/usr/src/cat.c.",
))


def main():
    OUT.mkdir(exist_ok=True)
    # small held-out val: last example; rest train (extended later)
    train = EXAMPLES[:-1]
    val = EXAMPLES[-1:]
    with open(OUT / "edit_discipline_train.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(OUT / "edit_discipline_val.jsonl", "w") as f:
        for e in val:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(train)} train + {len(val)} val examples to {OUT}")
    # sanity: longest example token-ish size
    longest = max(len(json.dumps(e)) for e in EXAMPLES)
    print(f"longest example ~{longest} chars (~{longest // 4} tokens)")


if __name__ == "__main__":
    main()
