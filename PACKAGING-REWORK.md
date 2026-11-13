# SPEAR packaging rework — a company commons base (build system included), then project deltas

Status: **draft for review**. No code changed yet.

## 1. Goal

The dividing line is **common across a company's projects vs specific to one
project** — not "small/public vs large/private".

- **The base image is a company's commons.** It carries what every project of
  that company needs: the shared **build system** (`infrabase` and its family),
  the cross-project corpora, the common skills and rules, the standards, and the
  common knowledge seed. `infrabase` belongs here — it underpins OpenCN, sye,
  pos and edgem1 alike — and so do the other corpora that are not tied to a
  single project.

- **Project content is the delta.** What is specific to one project
  (OpenCN's maths and algorithms, for instance) lives in that project's
  `spear-private` tree, **mounted on the base at run time** by default, and
  **optionally baked** into a project-pinned image later.

- **Multi-tenant.** A company X does the same: it builds its own base from X's
  commons (X's build system, X's corpora), then mounts or bakes X's projects.
  The base is what makes "a private company SPEAR image" reusable across all of
  that company's work.

Because the base carries the build system and private corpora, **the base is
private**. The only redistributable image is the content-free harness.

```
   spear:<ver>-harness      SPEAR app only — the one public, GitHub-shippable image
          |
   spear:<ver>-base         a company's commons: build system (infrabase) + common
     (per company)          corpora + skills + rules + standards + knowledge seed
          |                 (private; the "company SPEAR image")
   mount (default) or bake
     a project's spear-private
          |
   project-pinned image     optional: base + one project's spear-private baked in
```

## 2. Where we are today, and the gap

`scripts/docker/build.sh` has three **flat** profiles built independently from
one `docker/Dockerfile`: `public` (harness + only git-tracked content, no index,
no baked corpus), `reds` (corpora named by `--bake` + cut index, HEIG registry),
`private` (everything the host has). The reusable machinery already exists:

- **Per-corpus visibility** `"public": true` in `projects.json`
  (`gen-registry.py`); unmarked ⇒ private (fail-safe).
- **Per-document provenance** `source_origin` PUBLIC | LICENSED_STANDARD
  (`stage-standards.py`); unknown ⇒ licensed (fail-safe).
- **Commons vs per-user state already separated**: `/opt/spear/common`
  (`SPEAR_COMMON_STATE_DIR`, read-only seed) vs `/opt/spear/state`
  (`SPEAR_STATE_DIR`, writable, one run's history/memory/audit).
- **Baked trees land writable**: `COPY --from=baked --chmod=0777 . /corpora/`,
  shadowed by a bind mount when the host has the tree.
- **Named, lazy build contexts** keep the 4.3 GB embedder and venv cached.

The gap: nothing today builds a **company commons** image — harness + the shared
build system (`infrabase`) + common corpora + skills + rules + standards +
knowledge, as one reusable private image that projects then extend. `reds`/
`private` are flat, HEIG-specific (`ch.heig-vd.reds.spear.*`), and rebuild the
whole image each time instead of `base + delta`.

## 3. Target model

Three links in a `FROM` chain, so each layer is reused by the next:

| tier | name | FROM | carries | redistributable |
|---|---|---|---|---|
| 0 | `harness` | `ubuntu` | SPEAR app, venv, embedder, empty content dirs | **true** (public, GitHub) |
| 1 | `base` | `harness` | a company's commons: common corpora **and the build system `infrabase`** (writable + indexed), common skills, rules, standards, knowledge seed | **private** (carries private trees) |
| 2 | project image | `base` | base + one project's `spear-private` + project corpora, baked | false |

- **`base` is per company** and is the main deliverable — the private "company
  SPEAR image". It bakes the common corpora **including full build trees like
  `infrabase`** (`ib_so3`, `ib_u-boot`, `ib_avz`), writable under `/corpora` so
  the assistant can build and modify them in the container. Cross-toolchains
  stay **mounted** from the host (decision 6), so the base carries the build
  *tree*, the host provides the compiler.

  ```
  spear-image build base --entity edgemtech \
      --commons infrabase,ib_so3,ib_u-boot,ib_avz,<other common corpora> \
      [--knowledge <projects>]
  ```

- **Project content stays out by default.** A project's `spear-private`
  (OpenCN's maths, its project corpora, its rules/skills) is **mounted** on the
  base at run time via `SPEAR_CONTENT_TREES` / the `spear-private` mount, so it
  stays live and editable with no rebuild. Baking it is the optional tier-2 step
  for a self-contained project image:

  ```
  spear-image build project --from base --tree /path/to/opencn-ng/spear-private \
      [--bake <project corpora>] [--standards <store>] [--licensed]
  ```

  A project also adds its **own standards**: `--licensed --standards <store>`
  bakes the licensed documents of that store over the base's. So an edgemtech
  project image is the **reds base + a private standard** -- the base is one entity's
  commons, the project adds another's licensed standard, and the result is a
  non-redistributable project image owned by whoever built it.

- **Redistributability is derived, not a tier.** An image is `redistributable=
  true` only if every baked corpus is `public:true` and every standard is
  PUBLIC. A commons base that carries `infrabase` (private) is therefore private.
  A company whose commons happen to be all-public would get a public base; the
  guardrails decide, the tier does not.

## 4. Concrete changes, file by file

- **`docker/Dockerfile` → a FROM chain of three.**
  - `Dockerfile.harness`: ubuntu + apt + venv + embedder + the app code + empty
    `rules.d`/`skills`/`benches`/`corpora`. Public, content-free. This is today's
    Dockerfile minus the content COPYs.
  - `Dockerfile.base`: `FROM ${HARNESS_IMAGE}`, then bake the commons — the
    `index`, `rules`, `skills`, `benches`, `standards`, `common` (knowledge),
    the baked common corpora **including `infrabase`** into `/corpora`, the
    generated registry, and the base manifest.
  - `Dockerfile.project`: `FROM ${BASE_IMAGE}`, then COPY one project's delta
    (its `spear-private` content merged over the base, its project corpora into
    `/corpora`, its registry overlay).

- **`scripts/docker/build.sh`.** Add `--profile harness | base | project`.
  - `base`: bake the corpora named `--commons` (which may be full build trees
    like `infrabase`, `public:false`) + their index + common content + standards
    + knowledge; label `redistributable` by the derived rule and
    `entity=<name>`.
  - `project`: `FROM` the base tag, bake `--tree`'s content and `--bake` corpora.
  - Keep `public`/`reds`/`private` as thin aliases for one release (migration).

- **`gen-registry.py`.** A `base` registry = that company's commons corpora
  (common set, `infrabase` included); a `project` registry overlays the
  project's own corpora on top.

- **`stage-standards.py`.** The base takes the standards its company may carry
  (PUBLIC always; a company's licensed docs when it owns them); a project adds
  nothing new unless it owns standards of its own.

- **`stage-content.py` / `stage-corpora.py` / `stage-knowledge.py` /
  `stage-index.py`.** Unchanged logic, invoked once for the base commons and
  once for a project delta. `stage-corpora.py --bake` already copies a tree
  writable into `/corpora`; `stage-index.py --bake` cuts the index to a set.

- **`scripts/spear-image`.** New verbs: `build harness`, `build base --entity
  --commons [...]`, `build project --from base --tree [...]`. `run`/`save`/
  `load`/`push` unchanged; `push` still refuses a `redistributable=false` image
  without `--allow-push`.

- **Base content manifest.** `Dockerfile.base` bakes
  `/opt/spear/common/base-manifest.json` (baked corpus list incl. `infrabase`,
  index build date, standards manifest); `spear-image inspect base` prints it, a
  project image records the base tag it came from.

- **Labels.** Neutral `spear.*` namespace + `spear.entity=<name>`;
  `redistributable` governs `push`.

## 5. Redistributability guardrails

- A corpus is baked redistributable **only** if `projects.json` marks it
  `"public": true`; `infrabase` is `public:false`, so any image carrying it is
  `redistributable=false`. Unmarked ⇒ private.
- A standard travels into a redistributable image **only** if `source_origin` is
  PUBLIC; unknown ⇒ licensed ⇒ withheld.
- A run's own `state` (history, memories, audit) is **never** baked; only the
  common seed (`/opt/spear/common`) travels, read-only.

## 6. Layer reuse, distribution, migration

- **Reuse.** `harness` holds the expensive venv + embedder, shared by every
  base; a base holds the commons, shared by every project image. N projects of a
  company cost `base + N small deltas`.
- **Distribution.** A base builds `FROM harness`, a project `FROM base`; publish
  or `spear-image save`/`load` each. Tags are pinned to `spearversion.sh`.
- **Migration.** `public → harness`; `reds`/`private → base --entity <company>`
  (a company's commons, private); project-pinned images are the new tier 2. Keep
  the old words as aliases for one release.

## 7. Decisions taken (2026-11-10)

1. **Names.** `harness` (public, content-free) / `base` (a company's private
   commons) / `project` (optional, base + a baked project delta).
2. **The base carries the build system.** `infrabase` and its family are
   **commons**, baked into the base (trees **and** index, writable), not left to
   a per-project image. The base/delta line is "common across the company's
   projects vs specific to one project", and `infrabase` is common.
3. **A `FROM` chain**: `Dockerfile.harness` → `Dockerfile.base` →
   `Dockerfile.project`, so each tier reuses the one below.
4. **Labels** renamed to neutral `spear.*` + `spear.entity=<name>`.
5. **Base content manifest** baked into every base.
6. **Toolchains stay mounted**, not baked: the base carries the build *tree*
   (`infrabase`), the host provides the cross-toolchains at run time, as
   `spear-docker.sh` does today. (A later `--toolchains` flag could bake them for
   a fully standalone image; out of scope for now.)
7. **Redistributability is derived** from content, not a tier: a base carrying
   `infrabase` is private; only `harness` is public.

## 8. Implementation status (branch `spear-packaging-rework`)

A first implementation is on this branch, built to be **additive and safe**: the
legacy flat build (`build.sh`, `docker/Dockerfile`, `gen-registry.py`,
`stage-standards.py`) is **untouched**, so nothing existing breaks. The new tier
build lives in its own script.

Added / changed:

- `docker/Dockerfile.harness` — tier 0 (app only, empty content dirs, empty
  registry, public).
- `docker/Dockerfile.base` — `FROM ${HARNESS_IMAGE}`, bakes the commons
  (index, content, corpora incl. `infrabase`, standards, knowledge, manifest).
- `docker/Dockerfile.project` — `FROM ${BASE_IMAGE}`, bakes one project's delta:
  its content, corpora, index overlay, and **its own standards** appended over
  the base's (the reds-base + a private standard case).
- `scripts/docker/build-tiered.sh` — orchestrates `harness` / `base` /
  `project`: stages via the existing `stage-*` scripts (unchanged), maps the
  profile for `stage-standards.py` (base → public, or private with `--licensed`;
  project → private) so no Python needed a new profile value, derives
  `redistributable` from the baked corpora, and writes the manifest.
- `scripts/spear-image` — `build harness|base|project` dispatch to
  `build-tiered.sh`; `run`/`inspect`/`save` resolve the new tier names.
- `scripts/docker/push.sh` — decides on `redistributable` + `entity` instead of
  the profile name, reads the neutral `spear.*` labels (falling back to the old
  `ch.heig-vd.reds.spear.*`), and routes a private image to its entity's own
  registry (`SPEAR_<ENTITY>_REGISTRY`; `reds` → reds-gitlab, `edgemtech` →
  gitlab.edgemtech). A base/project headed for its entity's registry pushes
  without `--allow-push`; anywhere else still needs it. Legacy `reds`/`private`
  behaviour is preserved.

Verified: `bash -n` on all scripts, `--help`, the argument-validation paths, and
the manifest / redistributable-check Python in isolation. **Not yet run** is a
real `docker build` (it needs the configured build host — the one with
`client/machine.env`, `client/projects.json` and the index — e.g.
`/opt/llm/spear`). To validate end to end there:

```sh
spear-image build harness
spear-image build base --commons infrabase,ib_so3,ib_u-boot,ib_avz,<others> --entity edgemtech
spear-image build project --from spear:0.3.0-base --tree /path/to/opencn-ng/spear-private --bake <project corpora>
```

Still legacy-only and worth folding in next: a neutral `spear.*` label on the
old profiles too, and retiring `public`/`reds`/`private` once the tiers are
proven.
