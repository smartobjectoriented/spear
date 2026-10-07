"""A synthetic catalog of external capabilities, and labelled discovery cases.

Fifty tools across ten kinds of external system, each declared the way an MCP
server would: a name, a description, an input schema. The first ten hold the
answer to every case, so the 10-, 25- and 50-tool families are nested and a
larger one only adds distractors. Invented systems; nothing here is real.
"""

from __future__ import annotations


def _tool(name, description, properties, required=()):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object",
                            "properties": {key: {"type": kind, "description": text}
                                           for key, (kind, text) in properties.items()},
                            "required": list(required)}}


STR = "string"
INT = "integer"

TOOLS = [
    # The ten every case is answered from.
    _tool("issues_search", "Search the issue tracker by text, label or state.",
          {"query": (STR, "words to match"), "state": (STR, "open or closed")}, ["query"]),
    _tool("issues_create", "Open a new issue in the tracker.",
          {"title": (STR, "one-line summary"), "body": (STR, "details")}, ["title"]),
    _tool("ci_pipeline_status", "Result of the latest CI pipeline on a branch.",
          {"branch": (STR, "branch name")}, ["branch"]),
    _tool("ci_job_log", "Full log of one CI job, for reading why it failed.",
          {"job_id": (STR, "job identifier")}, ["job_id"]),
    _tool("artifact_latest", "Latest published build artifact of a component: version, URL.",
          {"component": (STR, "component name")}, ["component"]),
    _tool("wiki_page_read", "Read a page of the team wiki by its title.",
          {"title": (STR, "page title")}, ["title"]),
    _tool("deploy_status", "Which version is deployed in an environment.",
          {"environment": (STR, "staging or production")}, ["environment"]),
    _tool("metrics_query", "Time series of a service metric over a window.",
          {"metric": (STR, "metric name"), "window": (STR, "e.g. 1h, 24h")}, ["metric"]),
    _tool("chat_post_message", "Post a message to a team chat channel.",
          {"channel": (STR, "channel name"), "text": (STR, "message")}, ["channel", "text"]),
    _tool("review_request_list", "Open code review requests awaiting a reviewer.",
          {"repository": (STR, "repository name")}, ["repository"]),
    # Distractors, near and far.
    _tool("issues_comment", "Add a comment to an existing issue.",
          {"issue": (INT, "issue number"), "text": (STR, "comment")}, ["issue", "text"]),
    _tool("issues_close", "Close an issue with a resolution.",
          {"issue": (INT, "issue number"), "resolution": (STR, "why")}, ["issue"]),
    _tool("ci_pipeline_trigger", "Start a new CI pipeline on a branch.",
          {"branch": (STR, "branch name")}, ["branch"]),
    _tool("ci_job_retry", "Retry one failed CI job.", {"job_id": (STR, "job")}, ["job_id"]),
    _tool("artifact_list", "All published versions of a component.",
          {"component": (STR, "component name")}, ["component"]),
    _tool("wiki_page_search", "Search the team wiki by text.",
          {"query": (STR, "words")}, ["query"]),
    _tool("wiki_page_update", "Replace the content of a wiki page.",
          {"title": (STR, "page"), "content": (STR, "new content")}, ["title", "content"]),
    _tool("deploy_rollback", "Roll an environment back to its previous version.",
          {"environment": (STR, "environment")}, ["environment"]),
    _tool("metrics_alerts", "Alerts currently firing for a service.",
          {"service": (STR, "service")}, ["service"]),
    _tool("chat_read_channel", "Recent messages of a team chat channel.",
          {"channel": (STR, "channel"), "limit": (INT, "how many")}, ["channel"]),
    _tool("review_request_create", "Open a code review request for a branch.",
          {"repository": (STR, "repository"), "branch": (STR, "branch")},
          ["repository", "branch"]),
    _tool("calendar_free_slots", "Free time slots of a person in a date range.",
          {"person": (STR, "person"), "from": (STR, "date"), "to": (STR, "date")}, ["person"]),
    _tool("calendar_create_event", "Create a calendar event and invite people.",
          {"title": (STR, "title"), "when": (STR, "date and time"),
           "attendees": (STR, "comma-separated")}, ["title", "when"]),
    _tool("inventory_part_lookup", "Stock level and location of a hardware part.",
          {"part": (STR, "part number")}, ["part"]),
    _tool("inventory_reserve", "Reserve units of a hardware part for a project.",
          {"part": (STR, "part number"), "quantity": (INT, "units")}, ["part", "quantity"]),
] + [
    _tool(f"{area}_{verb}", f"{text}", {"name": (STR, "subject")}, ["name"])
    for area, verb, text in (
        ("license", "check", "Licence terms recorded for a third-party package."),
        ("license", "report", "Licence report of all dependencies of a product."),
        ("sbom", "export", "Export the software bill of materials of a release."),
        ("cve", "lookup", "Known vulnerabilities affecting a package version."),
        ("hw_lab", "book", "Book a board in the hardware lab for a time slot."),
        ("hw_lab", "status", "Which lab boards are free, booked or offline."),
        ("translation", "fetch", "Translated strings of a user-interface catalogue."),
        ("translation", "push", "Upload new source strings for translation."),
        ("support", "ticket_read", "Read a customer support ticket."),
        ("support", "ticket_reply", "Reply to a customer support ticket."),
        ("release", "notes_draft", "Draft release notes from merged changes."),
        ("release", "tag", "Create a release tag in a repository."),
        ("doc_site", "publish", "Publish the documentation site."),
        ("doc_site", "preview", "Build a preview of the documentation site."),
        ("access", "request", "Request access to a system for a person."),
        ("access", "audit", "Who has access to a system."),
        ("budget", "remaining", "Remaining budget of a project."),
        ("time", "log", "Log hours against a project."),
        ("time", "report", "Hours logged on a project over a period."),
        ("vendor", "contact", "Contact details of a component vendor."),
        ("vendor", "order_status", "Status of an order placed with a vendor."),
        ("energy", "usage", "Energy use of a test rack over a window."),
        ("backup", "status", "Last backup of a server and its result."),
        ("backup", "restore", "Restore a server from a backup."),
        ("dns", "lookup", "DNS records of an internal host name."))
]

CASES = [
    ("Is there already an issue about the UART driver dropping bytes?", "issues_search"),
    ("Open a ticket saying the bootloader hangs on cold start.", "issues_create"),
    ("Did the CI pass on branch feature/dma?", "ci_pipeline_status"),
    ("Why did CI job 4711 fail? Show me what it printed.", "ci_job_log"),
    ("What is the newest published build of the sensor-hub component?", "artifact_latest"),
    ("What does our wiki page 'Board bring-up' say about the clock setup?", "wiki_page_read"),
    ("Which firmware version is running in production right now?", "deploy_status"),
    ("How did request latency of the gateway service evolve over the last day?",
     "metrics_query"),
    ("Tell the #firmware channel that the nightly build is fixed.", "chat_post_message"),
    ("Which code reviews in the kernel repository are still waiting for someone?",
     "review_request_list"),
    ("Find issues mentioning a watchdog reset that are still open.", "issues_search"),
    ("What did the last pipeline on main report?", "ci_pipeline_status"),
]

assert len(TOOLS) == 50 and len({tool["name"] for tool in TOOLS}) == 50
assert all(target in {tool["name"] for tool in TOOLS[:10]} for _, target in CASES)
