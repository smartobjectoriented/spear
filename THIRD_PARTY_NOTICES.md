# Third-party notices

SPEAR is licensed under the Apache License 2.0 (see `LICENSE`). Parts of it
are copied or adapted from the projects below, under their own licenses,
whose notices are reproduced here as those licenses require. Source reuse
does not make SPEAR depend on these projects at runtime: none of them is
imported, installed or called.

## Hermes Agent

- Upstream: https://github.com/NousResearch/hermes-agent
- Revision: v0.21.0, commit 0cbc6e37
- License: MIT (repository root `LICENSE`; the files below carry no other
  license header)
- Copyright (c) 2025 Nous Research

| Upstream path | Use | SPEAR destination |
|---|---|---|
| `tools/fuzzy_match.py` | copied unmodified | `spear/agent/hermes/fuzzy_match.py` |
| `tools/ansi_strip.py` | copied unmodified | `spear/agent/hermes/ansi_strip.py` |
| `tools/binary_extensions.py` | copied unmodified | `spear/agent/hermes/binary_extensions.py` |
| `tools/terminal_hints.py` | copied unmodified | `spear/agent/hermes/terminal_hints.py` |
| `tools/terminal_tool.py` lines 2530-2670 (exit-code interpretation) | copied unmodified | `spear/agent/hermes/exit_codes.py` |
| `agent/tool_result_classification.py` | copied unmodified | `spear/agent/hermes/tool_result_classification.py` |
| `agent/tool_guardrails.py` | copied; imports adapted (`safe_json_loads` from `utils.py` inlined, package-relative import) | `spear/agent/hermes/tool_guardrails.py` |
| `agent/agent_runtime_helpers.py` `strip_think_blocks` and its patterns | copied; unused `agent` parameter dropped | `spear/agent/hermes/think.py` |
| `tools/file_operations.py` (`_detect_line_ending`, `_normalize_line_endings`, `_strip_bom`, `_MAGIC_SIGNATURES`, `identify_binary_bytes`, `describe_binary_file`, in-process linters, `_is_likely_binary_bytes`) and `tools/file_tools.py` (`_truncate_to_char_budget`, `_READ_DEDUP_STATUS_MESSAGE`, `_is_internal_file_*`) | copied unmodified (one staticmethod dedented) | `spear/agent/hermes/file_helpers.py` |
| `agent/prompt_builder.py` (`DEFAULT_AGENT_IDENTITY`, `TASK_COMPLETION_GUIDANCE`, `PARALLEL_TOOL_CALL_GUIDANCE`, `TOOL_USE_ENFORCEMENT_GUIDANCE`, `OPENAI_MODEL_EXECUTION_GUIDANCE`, `execution_guidance_text`), `agent/coding_context.py` (`CODING_AGENT_GUIDANCE`, the qwen edit-format line) | copied as evaluated | `spear/agent/hermes/prompt_text.py` |
| `agent/conversation_loop.py` (continuation, dropped-call, empty-response, invalid-JSON and truncation messages), `agent/context_compressor.py` (`MAX_ITERATIONS_SUMMARY_REQUEST`), `agent/agent_runtime_helpers.py` (`trailing_continue_intent`) | copied unmodified | `spear/agent/hermes/loop_constants.py` |
| `tools/file_tools.py` `read_file_tool`, `search_tool`, `patch_tool`, `write_file_tool`; `tools/file_operations.py` `read_file`, `search`, `_search_with_rg`, `_search_files_rg`, `_zero_match_probe`, `patch_replace`, `write_file`, `_check_lint_delta`, result classes; `tools/terminal_tool.py` result shaping; `tools/environments/base.py` `_wrap_command` and timeout semantics; `agent/coding_context.py` `_parse_status` (copied) and `build_coding_workspace_block` (git part) | ported function by function; deviations in the module docstring | `spear/agent/tools.py` |
| `agent/system_prompt.py` assembly order | ported; deviations in the module docstring | `spear/agent/prompt.py` |
| `agent/conversation_loop.py`, `run_agent.py` `_append_guardrail_observation`, `agent/chat_completion_helpers.py` message construction, `agent/transports/chat_completions.py` request shape, `agent/retry_utils.py` backoff | ported; deviations in the module docstrings | `spear/agent/loop.py`, `spear/agent/dispatch.py`, `spear/model_backend.py` (`complete_raw_messages`) |
| model-facing schemas of `read_file`, `search_files`, `patch`, `write_file`, `terminal` | copied, each trim listed in the file's `_edits` | `spear/agent/tool_schemas.json` |

`tools/fuzzy_match.py` states that its strategy chain was inspired by
OpenCode (https://github.com/sst/opencode, MIT); that is a design credit in
the upstream file, not code SPEAR took from OpenCode.

```
MIT License

Copyright (c) 2025 Nous Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## AnythingLLM

Inspected (https://github.com/Mintplex-Labs/anything-llm, commit 0713f551,
MIT, Copyright (c) Mintplex Labs Inc.; its `open-computer/` subproject is
AGPL-3.0 and was not used). No code was copied: SPEAR's tool selection is a
deterministic task-type → toolset mapping, not AnythingLLM's cross-encoder
reranker, which only activates above fifteen tools.
