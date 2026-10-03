"""Constants and predicates of Hermes Agent's conversation loop, copied.

From Hermes Agent (https://github.com/NousResearch/hermes-agent) at commit
0cbc6e37 (v0.21.0), unmodified: the synthetic continuation messages of
agent/conversation_loop.py and the trailing continue-intent stall guard of
agent/agent_runtime_helpers.py. Copyright (c) 2025 Nous Research. MIT
License -- see THIRD_PARTY_NOTICES.md at the repository root.

The intent-ack continuation (`_looks_like_codex_intermediate_ack`) is not
carried: with Hermes' default configuration it is "off" for every api_mode
but codex_responses, so it did not run for an OpenAI-compatible endpoint.
"""

import re

# agent/conversation_loop.py
_LENGTH_CONTINUATION_OUTPUT_LIMIT = (
    "[System: Your previous response was truncated by the output "
    "length limit. Continue exactly where you left off. Do not "
    "restart or repeat prior text. Finish the answer directly.]"
)
_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an "
    "empty response. Please process the tool "
    "results above and continue with the task."
)
_CODEX_ACK_CONTINUATION_NUDGE = (
    "[System: Continue now. Execute the required tool calls and only "
    "send your final answer after completing the task.]"
)


# agent/agent_runtime_helpers.py
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$",
    re.IGNORECASE,
)

# Content longer than this is a substantive reply, not a dangling ack.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 400


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action.

    Used by the stall-guard extension of the intent-ack continuation path in
    ``agent.conversation_loop``: when a turn is about to end with this shape
    (no tool calls, short content, trailing intent), the loop re-prompts via
    the existing bounded continuation mechanism instead of stopping.
    """
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    return bool(_TRAILING_CONTINUE_INTENT_RE.search(t[-160:]))


# agent/conversation_loop.py
_DROPPED_TOOLCALL_NUDGE_CONTENT = (
    "Your previous turn indicated a tool call but none was "
    "included. Do not narrate a plan or restate intent — issue "
    "the actual tool call now to continue the task."
)

# agent/context_compressor.py
MAX_ITERATIONS_SUMMARY_REQUEST = (
    "You've reached the maximum number of tool-calling iterations allowed. "
    "Please provide a final response summarizing what you've found and accomplished so far, "
    "without calling any more tools."
)

# agent/conversation_loop.py, inline strings of the invalid-arguments and
# truncation paths (lines 7753, 7812, 7852-7857).
INVALID_JSON_RESULT = (
    "Error: Invalid JSON arguments. {err}. "
    "For tools with no required parameters, use an empty object: {{}}. "
    "Please retry with valid JSON."
)
INVALID_JSON_SIBLING_RESULT = "Skipped: other tool call in this response had invalid JSON."
TRUNCATED_RESPONSE = "Response truncated due to output length limit"
