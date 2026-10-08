#!/usr/bin/env python3
"""Fetch ONE url and return it as text.

The assistant could search the web and not read it. `search_internet` returns
titles and snippets; the only way to open a result was to shell out to curl,
where the command policy refuses `network` in every mode but the ones the user
chose at launch. The visible failure was an answer that ended in "copy-paste
these URLs into your browser": the model had found the right NIST specification
and no way to read a line of it.

This runs IN the chat process, like the search it completes, and not in the
sandboxed shell. That is a deliberate split, not a hole: the capability policy
governs the arbitrary programs the model composes, while this is one fixed,
read-only operation whose boundary is written here —

  - http/https only. A `file://` or `gopher://` "url" is a local read wearing
    a network costume.
  - the destination must be a PUBLIC address, re-checked at EVERY redirect.
    This workstation tunnels :8082 to the inference server and sits on a
    /24 carrying other people's infrastructure; `http://127.0.0.1:8080/v1` or
    a 10.x host is not a web request, it is the model reaching into the
    network it happens to be running on.
  - a byte cap enforced WHILE streaming. Content-Length is a claim, and a
    30 GB body announced as 2 KB must stop at the cap, not at the surprise.
  - a character cap on what comes back, because the answer is spent from a
    65k context window and a specification is longer than that.
"""
import io
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

MAX_BYTES = 8 * 1024 * 1024

# Saving to disk is a different job from reading into a context window, and it
# has a different limit: a specification runs to tens of megabytes and a
# TRUNCATED download is not a smaller file, it is a corrupt one.
MAX_SAVE_BYTES = 64 * 1024 * 1024
MAX_CHARS = 40_000
MAX_PDF_PAGES = 60
MAX_REDIRECTS = 5
TIMEOUT = 20

# Named, and named honestly: a fetch that pretends to be a browser is a fetch
# whose refusals nobody can explain later.
USER_AGENT = "spear/1.0 (RAG assistant; +https://reds.heig-vd.ch)"

_TEXTUAL = ("text/", "application/json", "application/xml", "application/xhtml",
            "application/javascript", "+json", "+xml")


class FetchRefused(Exception):
    """Refused by this module's own boundary, before or during the request."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    content_type: str
    text: str
    truncated: bool
    note: str = ""
    #: For a PDF read in slices, the range that continues where this one
    #: stopped — "61-121", or None when there is nothing after it.
    next_pages: str | None = None

    def render(self):
        head = [f"# {self.final_url}"]

        if self.final_url != self.url:
            head.append(f"  (redirected from {self.url})")

        head.append(f"  {self.content_type}"
                    + (f" · {self.note}" if self.note else ""))

        if self.truncated:
            # Naming the way forward matters more here than anywhere: told only
            # that the answer was cut, the model re-fetched the SAME url five
            # times, then told the user to open a browser. Truncation is a
            # property of this tool, not of the document.

            head.append(f"  TRUNCATED at {MAX_CHARS} characters."
                        + (f" Read further with pages=\"{self.next_pages}\"."
                           if self.next_pages else
                           " Fetch a more specific page of the site."))

        return "\n".join(head) + "\n\n" + self.text


def _validate(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise FetchRefused(
            f"only http and https are fetched, not '{parsed.scheme or url}' — "
            f"a local path is read with the file tools, not with this one")

    if not parsed.hostname:
        raise FetchRefused(f"no host in {url!r}")

    return parsed


def _require_public(host, resolve=socket.getaddrinfo):
    """Every address `host` resolves to must be public.

    Checked per hop, because a redirect is a second destination chosen by the
    first one: a public page answering `302 http://10.0.0.5:8010/` would
    otherwise walk straight into the neighbours' infrastructure.
    """
    try:
        infos = resolve(host, None)
    except socket.gaierror as exc:
        raise FetchRefused(f"cannot resolve {host}: {exc}") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])

        if not address.is_global or address.is_multicast:
            raise FetchRefused(
                f"{host} resolves to {address}, which is not a public address. "
                f"fetch_url reaches the internet — not this machine, not the "
                f"LAN it is on, and not the inference server it is talking to.")


def _read_capped(response, cap=MAX_BYTES):
    """The body, and whether the cap cut it short."""
    chunks, size = [], 0

    for chunk in response.iter_content(64 * 1024):
        chunks.append(chunk)
        size += len(chunk)

        if size >= cap:
            response.close()

            return b"".join(chunks)[:cap], True

    return b"".join(chunks), False


def _html_to_text(data, encoding):
    from lxml import html as lxml_html

    try:
        document = lxml_html.fromstring(data)
    except Exception as exc:                       # malformed markup
        raise FetchRefused(f"cannot parse the HTML: {exc}") from exc

    # Script and style text is not page content; left in, a single minified
    # bundle fills the whole character budget with what nobody reads.

    for node in document.xpath("//script | //style | //noscript | //svg"):
        node.getparent().remove(node)

    return document.text_content()


def parse_pages(pages, total):
    """"61-121", "61-", "61" -> a 1-based inclusive (first, last) window.

    Bounded to MAX_PDF_PAGES so one call can never be the whole book, and
    clamped to what the document has rather than refused: asking for 61-200 of
    a 121-page PDF means "the rest".
    """
    if not pages:
        return 1, min(MAX_PDF_PAGES, total)

    text = str(pages).strip()
    found = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+)?)?", text)

    if not found:
        raise FetchRefused(
            f"pages={pages!r} is not a range — use \"61-121\", \"61-\" or \"61\".")

    first = max(1, int(found.group(1)))

    if first > total:
        raise FetchRefused(f"the document has {total} pages; {first} is past the end.")

    last = total if (found.group(2) is None and text.endswith("-")) else (
        int(found.group(2)) if found.group(2) else first)

    return first, min(total, max(first, last), first + MAX_PDF_PAGES - 1)


def _pdf_to_text(data, truncated, pages=None):
    if truncated:
        raise FetchRefused(
            f"the PDF is larger than the {MAX_BYTES // (1024 * 1024)} MB read "
            f"cap; a partial PDF cannot be parsed. Save it instead "
            f"(save_as=...), which allows "
            f"{MAX_SAVE_BYTES // (1024 * 1024)} MB.")

    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        total = len(reader.pages)
    except Exception as exc:
        raise FetchRefused(f"cannot read the PDF: {exc}") from exc

    first, last = parse_pages(pages, total)
    kept = reader.pages[first - 1:last]
    text = "\n\n".join((page.extract_text() or "") for page in kept)
    note = f"pages {first}-{last} of {total}"
    following = f"{last + 1}-{total}" if last < total else None

    if not text.strip():
        raise FetchRefused(
            f"{note} hold no extractable text — most likely scanned images, "
            f"which this tool does not OCR.")

    return text, note, following


def _tidy(text):
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _request(url, get, resolve, *, cap):
    """The validated, redirect-followed response and its capped body."""

    if get is None:
        import requests

        def get(target):
            return requests.get(target, stream=True, timeout=TIMEOUT,
                                allow_redirects=False,
                                headers={"User-Agent": USER_AGENT,
                                         "Accept": "*/*"})

    original, current = url, url

    for _ in range(MAX_REDIRECTS + 1):
        parsed = _validate(current)
        _require_public(parsed.hostname, resolve)

        try:
            response = get(current)
        except Exception as exc:                   # connection, TLS, timeout
            raise FetchRefused(f"{current}: {exc}") from exc

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")

            if not location:
                raise FetchRefused(
                    f"{current}: {response.status_code} with no Location")

            current = urljoin(current, location)

            continue

        break
    else:
        raise FetchRefused(f"more than {MAX_REDIRECTS} redirects from {original}")

    if response.status_code >= 400:
        raise FetchRefused(f"{current}: HTTP {response.status_code}")

    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
    data, capped = _read_capped(response, cap)

    return current, content_type, response.encoding or "utf-8", data, capped


def download(url, *, get=None, resolve=socket.getaddrinfo):
    """The raw bytes, for saving. Never a partial file.

    Reading a document into a context window and putting it on disk are
    different jobs. fetch() returns text and truncates by design; a download
    that truncates produces a file that opens in nothing, so a body that hits
    the cap is refused rather than written.
    """
    final_url, content_type, _encoding, data, capped = _request(
        url, get, resolve, cap=MAX_SAVE_BYTES)

    if capped:
        raise FetchRefused(
            f"{final_url} is larger than the "
            f"{MAX_SAVE_BYTES // (1024 * 1024)} MB cap. A truncated download "
            f"is a corrupt file, so nothing was written.")

    if not data:
        raise FetchRefused(f"{final_url} returned an empty body.")

    return data, (content_type or "application/octet-stream"), final_url


def fetch(url, *, max_chars=MAX_CHARS, pages=None, get=None,
          resolve=socket.getaddrinfo):
    """`url` as text, or FetchRefused. `get`/`resolve` are injected by tests."""
    current, content_type, encoding, data, capped = _request(
        url, get, resolve, cap=MAX_BYTES)
    note, following = "", None

    if content_type == "application/pdf" or current.lower().endswith(".pdf"):
        text, note, following = _pdf_to_text(data, capped, pages)
    elif any(marker in content_type for marker in ("html", "xhtml")):
        text = _html_to_text(data, encoding)
    elif not content_type or any(content_type.startswith(m) or m in content_type
                                 for m in _TEXTUAL):
        text = data.decode(encoding, errors="replace")
    else:
        raise FetchRefused(
            f"{current} is {content_type}, which has no text to return. To put "
            f"it on disk, pass save_as=<path>; this tool otherwise reads "
            f"pages and documents rather than downloading them.")

    text = _tidy(text)
    truncated = capped or len(text) > max_chars

    return FetchResult(url, current, content_type or "unknown",
                       text[:max_chars], truncated, note, following)
