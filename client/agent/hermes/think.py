"""strip_think_blocks and its patterns, copied.

From Hermes Agent (https://github.com/NousResearch/hermes-agent) at commit
0cbc6e37 (v0.21.0), agent/agent_runtime_helpers.py, unmodified except that
the unused `agent` parameter is dropped from strip_think_blocks. Hermes
credits the tool-call block patterns to openclaw/openclaw#67318.
Copyright (c) 2025 Nous Research. MIT License -- see THIRD_PARTY_NOTICES.md.
"""

import re


_REASONING_TAG_NAMES = ("think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought")

_TOOL_CALL_TAG_NAMES = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")

_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _REASONING_TAG_NAMES
)

_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)

_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
    r'<function\b[^>]*\bname\s*=[^>]*>'
    r'(?:(?:(?!</function>).)*)</function>',
    re.DOTALL | re.IGNORECASE,
)

_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf'(?:^|\n)[ \t]*<(?:{"|".join(_REASONING_TAG_NAMES)})\b[^>]*>.*$',
    re.DOTALL | re.IGNORECASE,
)

_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf'</?(?:{"|".join(_REASONING_TAG_NAMES)})>\s*',
    re.IGNORECASE,
)

_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf'</(?:{"|".join(_TOOL_CALL_TAG_NAMES)}|function)>\s*',
    re.IGNORECASE,
)


def strip_think_blocks(content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Handles four cases:
      1. Closed tag pairs (`` <think>… ``) — the common path when
         the provider emits complete reasoning blocks.
      2. Unterminated open tag at a block boundary (start of text or
         after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
         closing tag is dropped.  Everything from the open tag to end
         of string is stripped.  The block-boundary check mirrors
         ``gateway/stream_consumer.py``'s filter so models that mention
         `` <think>`` in prose aren't over-stripped.
      3. Stray orphan open/close tags that slip through.
      4. Tag variants: `` <think>``, ``<thinking>``, ``<reasoning>``,
         ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
         case-insensitive.

    Additionally strips standalone tool-call XML blocks that some open
    models (notably Gemma variants on OpenRouter) emit inside assistant
    content instead of via the structured ``tool_calls`` field:
      * ``<tool_call>…</tool_call>``
      * ``<tool_calls>…</tool_calls>``
      * ``<tool_result>…</tool_result>``
      * ``<function_call>…</function_call>``
      * ``<function_calls>…</function_calls>``
      * ``<function name="…">…</function>`` (Gemma style)
    Ported from openclaw/openclaw#67318. The ``<function>`` variant is
    boundary-gated (only strips when the tag sits at start-of-line or
    after punctuation and carries a ``name="..."`` attribute) so prose
    mentions like "Use <function> in JavaScript" are preserved.
    """
    if not content:
        return ""
    # Coerce non-string content to text before any regex runs.  Providers
    # that return assistant ``content`` as a list of blocks (Anthropic via
    # OpenRouter emits ``[{"type":"text",...}, {"type":"thinking",...}]``) or
    # as a dict flow into this shared helper from several callers — most
    # notably ``_interim_assistant_visible_text`` reading a *stored* history
    # message whose content was persisted as a list.  A raw list/dict reaching
    # ``re.sub`` below raises ``TypeError: expected string or bytes-like
    # object, got 'list'``, which the outer conversation loop swallows and
    # retries forever (observed as an infinite "preparing terminal…" loop on
    # Anthropic models via OpenRouter).  Flatten here so every caller is safe.
    if not isinstance(content, str):
        if isinstance(content, list):
            _parts: list[str] = []
            for _part in content:
                if isinstance(_part, str):
                    _parts.append(_part)
                elif isinstance(_part, dict):
                    _ptype = str(_part.get("type") or "").strip().lower()
                    # Drop reasoning/thinking blocks outright — this function's
                    # whole job is to strip them, and their text lives under
                    # different keys ("thinking", "reasoning") per provider.
                    if _ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    _text = _part.get("text")
                    if isinstance(_text, str) and _text:
                        _parts.append(_text)
            content = "".join(_parts)
        elif isinstance(content, dict):
            content = str(content.get("text") or content.get("content") or "")
        else:
            content = str(content)
        if not content:
            return ""
    # 1. Closed tag pairs — case-insensitive for all variants so
    #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
    #    the unterminated-tag pass and take trailing content with them.
    for _pattern in _REASONING_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
    #     generic tag names first — they have no attribute gating since
    #     a literal <tool_call> in prose is already vanishingly rare.
    for _pattern in _TOOL_CALL_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1c. <function name="...">...</function> — Gemma-style standalone
    #     tool call. Only strip when the tag sits at a block boundary
    #     (start of text, after a newline, or after sentence-ending
    #     punctuation) AND carries a name="..." attribute. This keeps
    #     prose mentions like "Use <function> to declare" safe.
    content = _NAMED_FUNCTION_BLOCK_PATTERN.sub('', content)
    # 2. Unterminated reasoning block — open tag at a block boundary
    #    (start of text, or after a newline) with no matching close.
    #    Strip from the tag to end of string.  Fixes #8878 / #9568
    #    (MiniMax M2.7 leaking raw reasoning into assistant content).
    content = _UNTERMINATED_REASONING_BLOCK_PATTERN.sub('', content)
    # 3. Stray orphan open/close tags that slipped through.
    content = _ORPHAN_REASONING_TAG_PATTERN.sub('', content)
    # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
    #     unterminated <function name="..."> because a truncated tail
    #     during streaming may still be valuable to the user; matches
    #     OpenClaw's intentional asymmetry.)
    content = _STRAY_TOOL_CALL_CLOSER_PATTERN.sub('', content)
    return content
