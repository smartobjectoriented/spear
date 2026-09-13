"""fetch_url's boundary — the part that is not "call requests".

The tool runs in the chat process, outside the sandbox that holds every other
network-capable thing, so the boundary IS the module. Each test below is one
of the ways a fetch stops being a web request.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web_fetch                                              # noqa: E402
from web_fetch import FetchRefused, fetch                     # noqa: E402


def resolver(address):
    """A getaddrinfo that answers `address` for any host."""
    family = 10 if ":" in address else 2

    return lambda host, port: [(family, 1, 6, "", (address, 0))]


PUBLIC = resolver("93.184.216.34")


def response(body=b"hi", status=200, content_type="text/plain", headers=None):
    reply = MagicMock()
    reply.status_code = status
    reply.headers = {"Content-Type": content_type, **(headers or {})}
    reply.encoding = "utf-8"
    reply.iter_content = lambda size: iter([body[i:i + size]
                                            for i in range(0, len(body), size)])
    return reply


class SchemeTests(unittest.TestCase):
    def test_only_http_and_https(self):
        for url in ("file:///etc/passwd", "gopher://x/1", "ftp://h/f",
                    "/home/operator/.ssh/id_rsa"):
            with self.assertRaises(FetchRefused) as raised:
                fetch(url, get=lambda _: response(), resolve=PUBLIC)

            self.assertIn("http", str(raised.exception))

    def test_the_refusal_points_at_the_file_tools(self):
        with self.assertRaises(FetchRefused) as raised:
            fetch("file:///etc/passwd", get=lambda _: response(), resolve=PUBLIC)

        self.assertIn("file tools", str(raised.exception))


class PrivateAddressTests(unittest.TestCase):
    """The machine tunnels :8082 to the inference server and sits on a /24 with
    other people's infrastructure. A name is not a destination — the ADDRESS
    it resolves to is, which is why this is checked after resolution."""

    def test_loopback_lan_and_link_local_are_refused(self):
        for address in ("127.0.0.1", "10.0.0.5", "192.168.1.10",
                        "169.254.169.254", "172.16.0.5", "::1", "fd00::1"):
            with self.assertRaises(FetchRefused) as raised:
                fetch("http://anything.example/", get=lambda _: response(),
                      resolve=resolver(address))

            self.assertIn("not a public address", str(raised.exception))

    def test_a_public_name_resolving_inward_is_still_refused(self):
        """The check is on the resolved address, so a public hostname pointed
        at 127.0.0.1 does not get through."""
        with self.assertRaises(FetchRefused):
            fetch("https://totally-legit.example/", get=lambda _: response(),
                  resolve=resolver("127.0.0.1"))

    def test_an_unresolvable_host_says_so(self):
        import socket

        def boom(host, port):
            raise socket.gaierror("Name or service not known")

        with self.assertRaises(FetchRefused) as raised:
            fetch("https://nope.invalid/", get=lambda _: response(), resolve=boom)

        self.assertIn("cannot resolve", str(raised.exception))

    def test_a_public_address_is_fetched(self):
        result = fetch("https://example.com/", get=lambda _: response(b"ok"),
                       resolve=PUBLIC)
        self.assertEqual("ok", result.text)


class RedirectTests(unittest.TestCase):
    def test_every_hop_is_checked_not_just_the_first(self):
        """A public page answering `302 http://10.0.0.5:8010/` would walk
        straight into the neighbours' infrastructure."""
        hops = {"https://example.com/":
                response(status=302, headers={"Location": "http://reds.internal/"})}

        def resolve(host, port):
            address = "127.0.0.1" if host == "reds.internal" else "93.184.216.34"

            return [(2, 1, 6, "", (address, 0))]

        with self.assertRaises(FetchRefused) as raised:
            fetch("https://example.com/",
                  get=lambda url: hops.get(url, response()), resolve=resolve)

        self.assertIn("not a public address", str(raised.exception))

    def test_a_redirect_chain_is_followed_and_the_final_url_reported(self):
        hops = {
            "https://a.example/": response(
                status=301, headers={"Location": "https://b.example/doc"}),
            "https://b.example/doc": response(b"arrived"),
        }
        result = fetch("https://a.example/", get=lambda url: hops[url],
                       resolve=PUBLIC)
        self.assertEqual("arrived", result.text)
        self.assertEqual("https://b.example/doc", result.final_url)
        self.assertIn("redirected from", result.render())

    def test_a_redirect_loop_ends(self):
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://a.example/",
                  get=lambda url: response(
                      status=302, headers={"Location": "https://a.example/"}),
                  resolve=PUBLIC)

        self.assertIn("redirects", str(raised.exception))

    def test_a_redirect_without_a_location_is_not_a_crash(self):
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://a.example/", get=lambda _: response(status=302),
                  resolve=PUBLIC)

        self.assertIn("no Location", str(raised.exception))


class BodyTests(unittest.TestCase):
    def test_the_byte_cap_is_enforced_while_streaming(self):
        """Content-Length is a claim. The cap has to hold against a body that
        keeps coming."""
        huge = b"x" * (web_fetch.MAX_BYTES + 5_000_000)
        result = fetch("https://example.com/", resolve=PUBLIC,
                       get=lambda _: response(huge, headers={"Content-Length": "12"}))
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.text), web_fetch.MAX_CHARS)

    def test_long_text_is_truncated_and_says_so(self):
        result = fetch("https://example.com/", resolve=PUBLIC,
                       get=lambda _: response(b"word " * 20_000))
        self.assertTrue(result.truncated)
        self.assertIn("TRUNCATED", result.render())

    def test_html_comes_back_without_its_script_and_style(self):
        page = (b"<html><head><style>.a{color:red}</style>"
                b"<script>var secret=1;</script></head>"
                b"<body><h1>Title</h1><p>Body text.</p></body></html>")
        result = fetch("https://example.com/", resolve=PUBLIC,
                       get=lambda _: response(page, content_type="text/html"))
        self.assertIn("Title", result.text)
        self.assertIn("Body text.", result.text)
        self.assertNotIn("secret", result.text)
        self.assertNotIn("color:red", result.text)

    def test_a_binary_type_is_refused_by_name(self):
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://example.com/x.zip", resolve=PUBLIC,
                  get=lambda _: response(b"PK\x03\x04",
                                         content_type="application/zip"))

        self.assertIn("application/zip", str(raised.exception))
        # …and it names the way to get it anyway.
        self.assertIn("save_as", str(raised.exception))

    def test_an_http_error_is_reported_not_parsed(self):
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://example.com/", resolve=PUBLIC,
                  get=lambda _: response(b"<h1>Not Found</h1>", status=404,
                                         content_type="text/html"))

        self.assertIn("404", str(raised.exception))


class PdfTests(unittest.TestCase):
    @staticmethod
    def _pdf(text="RS274NGC interpreter"):
        try:
            from pypdf import PdfWriter
        except ImportError:                                  # pragma: no cover
            raise unittest.SkipTest("pypdf is not installed")

        import io as _io
        from pypdf.generic import NameObject, create_string_object  # noqa: F401

        writer = PdfWriter()
        writer.add_blank_page(200, 200)
        buffer = _io.BytesIO()
        writer.write(buffer)

        return buffer.getvalue()

    def test_a_pdf_without_extractable_text_says_it_is_scanned(self):
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://example.com/spec.pdf", resolve=PUBLIC,
                  get=lambda _: response(self._pdf(),
                                         content_type="application/pdf"))

        self.assertIn("OCR", str(raised.exception))

    def test_a_truncated_pdf_is_refused_rather_than_half_parsed(self):
        huge = b"%PDF-1.4" + b"x" * (web_fetch.MAX_BYTES + 1000)
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://example.com/spec.pdf", resolve=PUBLIC,
                  get=lambda _: response(huge, content_type="application/pdf"))

        self.assertIn("cap", str(raised.exception))

    def test_a_pdf_is_recognised_by_extension_too(self):
        """Servers announce application/octet-stream for PDFs often enough."""
        with self.assertRaises(FetchRefused) as raised:
            fetch("https://example.com/spec.pdf", resolve=PUBLIC,
                  get=lambda _: response(self._pdf(),
                                         content_type="application/octet-stream"))

        self.assertIn("OCR", str(raised.exception))


class OfflineSessionTests(unittest.TestCase):
    """--no-network has to mean the session, not only the shell.

    search_internet and fetch_url browse from the chat process, outside the
    capability policy: a flag that only emptied `network` out of the execution
    modes would have read as offline without being it.
    """

    def _rag_chat(self, argv):
        import importlib
        import rag_chat

        saved = sys.argv
        sys.argv = ["rag_chat.py", *argv]
        try:
            return importlib.reload(rag_chat)
        finally:
            sys.argv = saved

    def tearDown(self):
        self._rag_chat([])                       # leave the module as found

    def test_no_network_hides_both_web_tools(self):
        rag_chat = self._rag_chat(["--no-network"])
        names = [tool["function"]["name"] for tool in rag_chat.TOOLS]
        self.assertNotIn("search_internet", names)
        self.assertNotIn("fetch_url", names)
        self.assertIn("search_corpus", names)    # retrieval is local, kept

    def test_the_handler_refuses_even_if_the_name_arrives_anyway(self):
        rag_chat = self._rag_chat(["--no-network"])
        result = rag_chat._registered_fetch_url(
            None, {"url": "https://example.com/"})
        self.assertIn("--no-network", str(result.text))

    def test_without_the_flag_both_are_exposed(self):
        rag_chat = self._rag_chat([])
        names = [tool["function"]["name"] for tool in rag_chat.TOOLS]
        self.assertIn("search_internet", names)
        self.assertIn("fetch_url", names)


class PageRangeTests(unittest.TestCase):
    """A long PDF is read in slices, and each slice says where the next starts.

    Told only that the answer was truncated, the model re-fetched the SAME url
    five times and then told the user to open a browser. Truncation is a
    property of this tool, not of the document.
    """

    def test_a_range_is_parsed_and_clamped_to_the_document(self):
        self.assertEqual((1, 60), web_fetch.parse_pages(None, 121))
        self.assertEqual((61, 120), web_fetch.parse_pages("61-121", 121))
        # "to the end" is still bounded by the per-call page cap
        self.assertEqual((61, 120), web_fetch.parse_pages("61-", 121))
        self.assertEqual((61, 61), web_fetch.parse_pages("61", 121))
        self.assertEqual((1, 3), web_fetch.parse_pages(None, 3))

    def test_a_window_never_exceeds_the_page_cap(self):
        first, last = web_fetch.parse_pages("1-10000", 10000)
        self.assertEqual(web_fetch.MAX_PDF_PAGES, last - first + 1)

    def test_a_start_past_the_end_is_refused_by_name(self):
        with self.assertRaises(FetchRefused) as raised:
            web_fetch.parse_pages("400", 121)

        self.assertIn("121 pages", str(raised.exception))

    def test_nonsense_says_what_a_range_looks_like(self):
        with self.assertRaises(FetchRefused) as raised:
            web_fetch.parse_pages("chapter two", 121)

        self.assertIn("61-121", str(raised.exception))

    def test_the_truncation_note_names_the_next_range(self):
        result = web_fetch.FetchResult(
            "u", "u", "application/pdf", "text", True, "pages 1-60 of 121",
            "61-121")
        self.assertIn('pages="61-121"', result.render())

    def test_html_truncation_does_not_promise_a_page_range(self):
        result = web_fetch.FetchResult("u", "u", "text/html", "t", True)
        self.assertNotIn("pages=", result.render())


class DownloadTests(unittest.TestCase):
    """Putting a file on disk is a different job from reading it."""

    def test_a_truncated_download_is_refused_rather_than_written(self):
        """A partial download is not a smaller file, it is a corrupt one."""
        huge = b"x" * (web_fetch.MAX_SAVE_BYTES + 1024)
        with self.assertRaises(FetchRefused) as raised:
            web_fetch.download("https://example.com/big.pdf", resolve=PUBLIC,
                               get=lambda _: response(huge, content_type="application/pdf"))

        self.assertIn("corrupt", str(raised.exception))

    def test_an_empty_body_is_refused(self):
        with self.assertRaises(FetchRefused):
            web_fetch.download("https://example.com/x.pdf", resolve=PUBLIC,
                               get=lambda _: response(b""))

    def test_bytes_come_back_untouched_with_their_type(self):
        data, content_type, final = web_fetch.download(
            "https://example.com/x.pdf", resolve=PUBLIC,
            get=lambda _: response(b"%PDF-1.2 body", content_type="application/pdf"))
        self.assertEqual(b"%PDF-1.2 body", data)
        self.assertEqual("application/pdf", content_type)
        self.assertEqual("https://example.com/x.pdf", final)

    def test_a_download_obeys_the_same_address_boundary(self):
        with self.assertRaises(FetchRefused) as raised:
            web_fetch.download("http://reds.internal/x.pdf",
                               resolve=resolver("10.0.0.5"),
                               get=lambda _: response(b"x"))

        self.assertIn("not a public address", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
