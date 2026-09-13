"""The embedding wire: both readers, held to the same bytes.

The protocol is defined once in `server/embed/protocol.py` and implemented
twice -- there and in `spear/embedding.py`. That is not an oversight. The
worker is installed on a different machine, with only `server/embed/` copied
to it and none of the client tree, so neither side can import the other even
if the boundary allowed it (`test_client_server_boundary`).

Two implementations of one format is exactly the situation that needs a
contract test, and this is it. It may import both sides because a test that
validates a contract has to look at both.

The failure it exists to prevent is the quiet one. A client and a worker that
still parse each other's bytes while disagreeing about who applies the
document prefix produce vectors that are subtly wrong and perfectly
well-formed, and a collection filled with them reports nothing -- retrieval
merely gets worse.
"""

from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "server" / "embed"))

import embedding                                             # noqa: E402
import protocol                                              # noqa: E402

TARGET = "operator@gpu-host.example"

WELL_FORMED = dict(model="BAAI/bge-m3", texts=["one", "two"],
                   prefix="passage: ", max_seq_length=1024, normalize=True,
                   batch_size=16, trust_remote_code=False)


def ok_response(vectors):
    """A worker's successful answer, built by the SERVER's writer."""
    flat = [value for vector in vectors for value in vector]

    return (protocol.encode_response_header(len(vectors), len(vectors[0]))
            + struct.pack(f"<{len(flat)}f", *flat))


class TheTwoSidesAgree(unittest.TestCase):
    """The contract, in both directions."""

    def test_the_clients_request_is_what_the_server_accepts(self):
        semantics = embedding.document_semantics("BAAI/bge-m3", 16)
        raw = embedding._protocol_request(["one", "two"], semantics)
        decoded = protocol.decode_request(raw)

        self.assertEqual(decoded["texts"], ["one", "two"])

        for field, value in semantics.items():
            with self.subTest(field=field):
                self.assertEqual(decoded[field], value)

    def test_the_clients_request_carries_every_field_the_server_requires(self):
        """A field the client forgets must not be filled in by the server."""
        semantics = embedding.document_semantics("BAAI/bge-m3", 16)
        raw = embedding._protocol_request(["one"], semantics)

        for field in json.loads(raw):
            trimmed = {k: v for k, v in json.loads(raw).items() if k != field}

            with self.subTest(dropped=field):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.decode_request(json.dumps(trimmed).encode())

    def test_the_servers_response_is_what_the_client_reads(self):
        vectors = [[0.5, -0.25, 0.125, 0.0], [1.0, 2.0, 3.0, 4.0]]
        read = embedding._protocol_response(ok_response(vectors), 2, TARGET)

        self.assertEqual(read, vectors)

    def test_both_readers_agree_on_the_same_bytes(self):
        vectors = [[0.5, -0.25, 0.125, 0.0]]
        raw = ok_response(vectors)
        header, body = protocol.decode_response(raw)

        self.assertEqual(protocol.unpack_vectors(header, body),
                         embedding._protocol_response(raw, 1, TARGET))

    def test_the_version_constant_is_the_same_number_on_both_sides(self):
        self.assertEqual(embedding.EMBED_PROTOCOL_VERSION,
                         protocol.PROTOCOL_VERSION)
        self.assertEqual(embedding.PROTOCOL_KEY, protocol.VERSION_KEY)


class AVersionMismatchIsRefusedLoudly(unittest.TestCase):
    """Never negotiated down. A pair that disagrees stops."""

    def test_the_server_refuses_a_future_client(self):
        raw = json.dumps(dict(WELL_FORMED,
                              **{protocol.VERSION_KEY: 99})).encode()

        with self.assertRaises(protocol.ProtocolError) as raised:
            protocol.decode_request(raw)

        message = str(raised.exception)
        self.assertIn("99", message)
        self.assertIn(str(protocol.PROTOCOL_VERSION), message)
        self.assertIn("version mismatch", message)

    def test_the_server_refuses_a_request_with_no_version_at_all(self):
        raw = json.dumps(WELL_FORMED).encode()

        with self.assertRaises(protocol.ProtocolError) as raised:
            protocol.decode_request(raw)

        self.assertIn(protocol.VERSION_KEY, str(raised.exception))

    def test_the_client_refuses_a_future_worker(self):
        header = json.dumps({protocol.VERSION_KEY: 99, "status": "ok",
                             "count": 1, "dim": 4, "byte_count": 16}) + "\n"
        raw = header.encode() + b"\0" * 16

        with self.assertRaises(embedding.RemoteEmbeddingError) as raised:
            embedding._protocol_response(raw, 1, TARGET)

        message = str(raised.exception)
        self.assertIn("99", message)
        self.assertIn("version mismatch", message)
        self.assertIn(TARGET, message)

    def test_the_client_refuses_a_response_with_no_version(self):
        raw = (json.dumps({"status": "ok", "count": 1, "dim": 4,
                           "byte_count": 16}) + "\n").encode() + b"\0" * 16

        with self.assertRaises(embedding.RemoteEmbeddingError):
            embedding._protocol_response(raw, 1, TARGET)


class MalformedRequestsAreRefused(unittest.TestCase):
    """Each of these is a way a vector could be silently wrong instead."""

    def refuse(self, **overrides):
        raw = json.dumps(dict(WELL_FORMED,
                              **{protocol.VERSION_KEY: protocol.PROTOCOL_VERSION},
                              **overrides)).encode()

        with self.assertRaises(protocol.ProtocolError) as raised:
            protocol.decode_request(raw)

        return str(raised.exception)

    def test_not_json_at_all(self):
        with self.assertRaises(protocol.ProtocolError) as raised:
            protocol.decode_request(b"not json")

        self.assertIn("not JSON", str(raised.exception))

    def test_a_json_array_is_not_a_request(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_request(b"[1, 2, 3]")

    def test_an_empty_model(self):
        self.assertIn("model", self.refuse(model="   "))

    def test_texts_that_are_not_strings(self):
        self.assertIn("texts[1]", self.refuse(texts=["ok", 7]))

    def test_no_texts_at_all(self):
        self.assertIn("nothing to encode", self.refuse(texts=[]))

    def test_a_prefix_that_is_not_a_string(self):
        self.assertIn("prefix", self.refuse(prefix=None))

    def test_a_normalize_that_is_a_number(self):
        """`normalize: 1` is a client that meant True and a server that must
        not guess which."""
        self.assertIn("normalize", self.refuse(normalize=1))

    def test_a_batch_size_that_is_a_boolean(self):
        self.assertIn("batch_size", self.refuse(batch_size=True))

    def test_a_batch_size_of_zero(self):
        self.assertIn("batch_size", self.refuse(batch_size=0))

    def test_a_negative_sequence_cap(self):
        self.assertIn("max_seq_length", self.refuse(max_seq_length=-1))

    def test_a_null_sequence_cap_is_allowed(self):
        """null means the model's own window, which is a real choice."""
        raw = json.dumps(dict(WELL_FORMED, max_seq_length=None,
                              **{protocol.VERSION_KEY: 1})).encode()

        self.assertIsNone(protocol.decode_request(raw)["max_seq_length"])


class TheClientRefusesAnythingThatIsNotAWorker(unittest.TestCase):
    """Every one of these has filled a collection with nonsense before."""

    def refuse(self, raw, expected=1):
        with self.assertRaises(embedding.RemoteEmbeddingError) as raised:
            embedding._protocol_response(raw, expected, TARGET)

        return str(raised.exception)

    def test_output_with_no_header_line(self):
        self.assertIn("no protocol header", self.refuse(b"hello"))

    def test_a_header_line_that_is_not_json(self):
        self.assertIn("no protocol header", self.refuse(b"1 4\n" + b"\0" * 16))

    def test_a_structured_error_is_reported_in_the_workers_own_words(self):
        raw = protocol.encode_error("CUDA out of memory",
                                    protocol.KIND_ENCODING)
        message = self.refuse(raw)

        self.assertIn("CUDA out of memory", message)
        self.assertIn("encoding", message)

    def test_a_truncated_body(self):
        vectors = [[0.5, -0.25, 0.125, 0.0]]
        self.assertIn("out of protocol", self.refuse(ok_response(vectors)[:-4]))

    def test_a_short_count_is_refused_rather_than_padded(self):
        """Two texts in, one vector back. Accepting that would misalign every
        chunk in the collection from there on."""
        self.assertIn("out of protocol",
                      self.refuse(ok_response([[1.0, 2.0]]), expected=2))


class TheClientOwnsTheSemantics(unittest.TestCase):
    """What travels is what the client decided, for the model it names."""

    def test_the_document_prefix_is_the_registry_entry(self):
        for model, (document, _query, _trust) in embedding.MODELS.items():
            if model == embedding.DEFAULT:
                continue

            with self.subTest(model=model):
                self.assertEqual(
                    embedding.document_semantics(model, 16)["prefix"], document)

    def test_an_asymmetric_model_sends_the_document_side_not_the_query_side(self):
        """e5 prefixes are asymmetric; sending the query one to index a
        corpus is several points of recall, silently."""
        model = "intfloat/multilingual-e5-large"
        document, query, _ = embedding.MODELS[model]

        self.assertNotEqual(document, query)
        self.assertEqual(embedding.document_semantics(model, 16)["prefix"],
                         document)

    def test_the_cap_and_the_normalisation_travel_explicitly(self):
        semantics = embedding.document_semantics("BAAI/bge-m3", 16)

        self.assertEqual(semantics["max_seq_length"], embedding.MAX_SEQ)
        self.assertIs(semantics["normalize"], True)

    def test_trust_remote_code_is_the_registry_entry_not_a_default(self):
        for model in embedding.MODELS:
            if model == embedding.DEFAULT:
                continue

            with self.subTest(model=model):
                self.assertEqual(
                    embedding.document_semantics(model, 16)["trust_remote_code"],
                    embedding.MODELS[model][2])

    def test_the_batch_size_the_caller_asked_for_is_the_one_sent(self):
        self.assertEqual(
            embedding.document_semantics("BAAI/bge-m3", 8)["batch_size"], 8)

    def test_the_local_path_applies_the_same_semantics(self):
        """Local and remote must be the same encoding on a different machine.

        Asserted on the call the encoder actually receives, because that is
        where a divergence would live -- not in the two functions' prose.
        """
        seen = {}

        class Encoder:
            def encode(self, texts, **kw):
                import numpy

                seen["texts"], seen["kw"] = texts, kw

                return numpy.zeros((len(texts), 4), dtype="float32")

        model = "intfloat/multilingual-e5-large"
        semantics = embedding.document_semantics(model, 8)

        with patch.object(embedding, "_st", return_value=Encoder()), \
             patch.object(embedding, "remote_target", return_value=None):
            embedding.embed_documents(["one"], model=model, batch_size=8)

        self.assertEqual(seen["texts"], [semantics["prefix"] + "one"])
        self.assertEqual(seen["kw"]["batch_size"], semantics["batch_size"])
        self.assertEqual(seen["kw"]["normalize_embeddings"],
                         semantics["normalize"])


class TheProtocolModuleNeedsNothingInstalled(unittest.TestCase):
    """It must be readable on a machine with no ML stack at all."""

    def test_it_imports_only_the_standard_library(self):
        import ast

        source = (REPO / "server" / "embed" / "protocol.py").read_text()
        imported = set()

        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        self.assertEqual(imported - {"json", "struct", "__future__"}, set())


if __name__ == "__main__":
    unittest.main()
