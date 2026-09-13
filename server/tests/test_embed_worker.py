"""The worker executes a request. It does not interpret one.

Everything here runs without torch and without weights: the encoder is
injected, so what is under test is the worker's own behaviour -- what it
passes through, what it refuses, what it decides for itself -- rather than
sentence-transformers.

The property that matters most is negative. The worker must have NO opinion
about prefixes, caps or normalisation. The worker this replaces had all three,
because it imported the client's module and read the client's registry on the
GPU host; two copies of that file, nothing comparing them, and a collection
filled with two different prefixes says nothing about it.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER / "embed"))

import protocol                                              # noqa: E402
import worker                                                # noqa: E402

REQUEST = dict(model="some/model", texts=["alpha", "beta"], prefix="passage: ",
               max_seq_length=1024, normalize=True, batch_size=16,
               trust_remote_code=False)


class Encoder:
    """A stand-in that records exactly what it was asked to do."""

    def __init__(self, dim=4, max_seq_length=8192):
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.calls = []

    def encode(self, texts, **kw):
        import numpy

        self.calls.append((list(texts), kw))

        return numpy.arange(len(texts) * self.dim,
                            dtype="float32").reshape(len(texts), self.dim)


def request_bytes(**overrides):
    body = dict(REQUEST, **overrides)
    body[protocol.VERSION_KEY] = protocol.PROTOCOL_VERSION

    return json.dumps(body).encode()


def run(raw, encoder=None, device="cpu"):
    """Drive the worker end to end; return (status, header, body, encoder)."""
    encoder = encoder or Encoder()
    out = io.BytesIO()

    with patch.object(worker, "load", return_value=encoder):
        status = worker.run(io.BytesIO(raw), out, io.StringIO(), device=device)

    raw_out = out.getvalue()
    head, _, body = raw_out.partition(b"\n")

    return status, json.loads(head), body, encoder


class ItAppliesTheRequestVerbatim(unittest.TestCase):
    def test_the_prefix_is_prepended_to_every_text(self):
        _status, _header, _body, encoder = run(request_bytes())

        self.assertEqual(encoder.calls[0][0],
                         ["passage: alpha", "passage: beta"])

    def test_an_empty_prefix_prepends_nothing(self):
        _s, _h, _b, encoder = run(request_bytes(prefix=""))

        self.assertEqual(encoder.calls[0][0], ["alpha", "beta"])

    def test_a_query_prefix_is_applied_exactly_as_sent(self):
        """The worker cannot tell a query from a document, and must not try.

        Which side of the asymmetry a prefix belongs to is the client's
        decision; the worker's job is to prepend the string it was given.
        """
        instruction = "Instruct: retrieve the relevant source\nQuery: "
        _s, _h, _b, encoder = run(request_bytes(prefix=instruction,
                                                texts=["where is it"]))

        self.assertEqual(encoder.calls[0][0], [instruction + "where is it"])

    def test_normalisation_is_passed_through_when_on(self):
        _s, _h, _b, encoder = run(request_bytes(normalize=True))

        self.assertIs(encoder.calls[0][1]["normalize_embeddings"], True)

    def test_normalisation_is_passed_through_when_off(self):
        """Off is a real choice. A worker that normalised anyway would make
        every un-normalised deployment silently wrong."""
        _s, _h, _b, encoder = run(request_bytes(normalize=False))

        self.assertIs(encoder.calls[0][1]["normalize_embeddings"], False)

    def test_the_batch_size_is_the_one_requested(self):
        _s, _h, _b, encoder = run(request_bytes(batch_size=3))

        self.assertEqual(encoder.calls[0][1]["batch_size"], 3)

    def test_it_never_narrates(self):
        """Progress output on a pipe carrying float32 would corrupt it."""
        _s, _h, _b, encoder = run(request_bytes())

        self.assertIs(encoder.calls[0][1]["show_progress_bar"], False)


class TheSequenceCapBoundsAndNeverWidens(unittest.TestCase):
    def test_a_wide_model_is_capped_to_the_request(self):
        encoder = Encoder(max_seq_length=8192)
        worker.apply_sequence_cap(encoder, 1024)

        self.assertEqual(encoder.max_seq_length, 1024)

    def test_a_narrow_model_keeps_its_own_limit(self):
        """min, not assignment: widening a window is not a cap."""
        encoder = Encoder(max_seq_length=512)
        worker.apply_sequence_cap(encoder, 1024)

        self.assertEqual(encoder.max_seq_length, 512)

    def test_a_null_cap_leaves_the_model_alone(self):
        encoder = Encoder(max_seq_length=8192)
        worker.apply_sequence_cap(encoder, None)

        self.assertEqual(encoder.max_seq_length, 8192)

    def test_a_model_that_reports_nothing_takes_the_request(self):
        encoder = Encoder(max_seq_length=None)
        worker.apply_sequence_cap(encoder, 1024)

        self.assertEqual(encoder.max_seq_length, 1024)

    def test_the_cap_is_applied_on_the_request_path(self):
        encoder = Encoder(max_seq_length=8192)
        run(request_bytes(max_seq_length=256), encoder)

        self.assertEqual(encoder.max_seq_length, 256)


class TheMachineDecidesTheDeviceAndNothingElse(unittest.TestCase):
    def test_an_explicit_choice_is_never_second_guessed(self):
        self.assertEqual(
            worker.select_device({"SPEAR_EMBED_DEVICE": "cpu"},
                                 cuda_available=True), "cpu")

    def test_a_card_is_used_when_there_is_one(self):
        self.assertEqual(worker.select_device({}, cuda_available=True), "cuda")

    def test_without_a_card_it_is_the_cpu(self):
        self.assertEqual(worker.select_device({}, cuda_available=False), "cpu")

    def test_an_empty_variable_is_not_a_device_name(self):
        self.assertEqual(
            worker.select_device({"SPEAR_EMBED_DEVICE": "  "},
                                 cuda_available=False), "cpu")

    def test_a_card_is_pinned_by_uuid(self):
        env = {"SPEAR_GPU_UUID": "GPU-00000000-0000-0000-0000-000000000000"}

        self.assertEqual(worker.pin_gpu(env), env["SPEAR_GPU_UUID"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], env["SPEAR_GPU_UUID"])

    def test_an_existing_choice_is_left_alone(self):
        env = {"CUDA_VISIBLE_DEVICES": "1",
               "SPEAR_GPU_UUID": "GPU-00000000-0000-0000-0000-000000000000"}

        self.assertIsNone(worker.pin_gpu(env))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")


class ItAnswersInProtocol(unittest.TestCase):
    def test_a_good_request_returns_a_conforming_response(self):
        status, header, body, _ = run(request_bytes())

        self.assertEqual(status, protocol.EXIT_OK)
        self.assertEqual(header["status"], "ok")
        self.assertEqual(header[protocol.VERSION_KEY],
                         protocol.PROTOCOL_VERSION)
        self.assertEqual((header["count"], header["dim"]), (2, 4))
        self.assertEqual(header["dtype"], "float32")
        self.assertEqual(len(body), header["byte_count"])

    def test_the_vectors_survive_the_round_trip(self):
        _status, header, body, _ = run(request_bytes())

        self.assertEqual(protocol.unpack_vectors(header, body),
                         [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])

    def test_a_version_mismatch_is_a_protocol_failure(self):
        raw = json.dumps(dict(REQUEST,
                              **{protocol.VERSION_KEY: 99})).encode()
        status, header, body, _ = run(raw)

        self.assertEqual(status, protocol.EXIT_PROTOCOL)
        self.assertEqual(header["kind"], protocol.KIND_PROTOCOL)
        self.assertIn("version mismatch", header["error"])
        self.assertEqual(body, b"")

    def test_a_malformed_request_is_a_protocol_failure(self):
        status, header, _b, _ = run(b"{not json")

        self.assertEqual(status, protocol.EXIT_PROTOCOL)
        self.assertEqual(header["status"], "error")
        self.assertEqual(header["kind"], protocol.KIND_PROTOCOL)

    def test_a_failure_to_encode_is_an_operational_failure(self):
        """Separated from a protocol failure on purpose: one means retry
        elsewhere, the other means no retry will ever help."""
        class Broken(Encoder):
            def encode(self, texts, **kw):
                raise RuntimeError("CUDA out of memory")

        status, header, _b, _ = run(request_bytes(), Broken())

        self.assertEqual(status, protocol.EXIT_ENCODING)
        self.assertEqual(header["kind"], protocol.KIND_ENCODING)
        self.assertIn("CUDA out of memory", header["error"])

    def test_an_encoder_returning_the_wrong_count_is_caught_here(self):
        """Better to fail on this machine than to misalign a collection on
        the other one."""
        class Short(Encoder):
            def encode(self, texts, **kw):
                import numpy

                return numpy.zeros((1, 4), dtype="float32")

        status, header, _b, _ = run(request_bytes(), Short())

        self.assertEqual(status, protocol.EXIT_ENCODING)
        self.assertIn("for 2 texts", header["error"])

    def test_a_model_that_cannot_be_loaded_is_reported_structurally(self):
        out = io.BytesIO()

        with patch.object(worker, "load",
                          side_effect=protocol.EncodingError("no such model")):
            status = worker.run(io.BytesIO(request_bytes()), out,
                                io.StringIO(), device="cpu")

        header = json.loads(out.getvalue().partition(b"\n")[0])

        self.assertEqual(status, protocol.EXIT_ENCODING)
        self.assertIn("no such model", header["error"])


class ItHasNoOpinionAboutRetrieval(unittest.TestCase):
    """The negative property, read off the source.

    A registry, a default prefix or a per-model branch here would let the two
    sides disagree about an encoding while both believed they were right.
    """

    def source(self):
        return (SERVER / "embed" / "worker.py").read_text(encoding="utf-8")

    def test_it_names_no_embedding_model(self):
        for model in ("bge-m3", "multilingual-e5", "Qwen3-Embedding",
                      "MiniLM"):
            with self.subTest(model=model):
                self.assertNotIn(model, self.source())

    def test_it_carries_no_prefix_literal(self):
        for prefix in ("passage: ", "query: ", "Instruct:"):
            with self.subTest(prefix=prefix):
                self.assertNotIn(prefix, self.source())

    def test_it_does_not_import_the_client(self):
        source = self.source()

        self.assertNotIn("import embedding", source)
        self.assertNotIn("sys.path.insert", source)

    def test_it_defaults_none_of_the_semantic_fields(self):
        """Every one arrives in the request or the request is refused."""
        for field in ("prefix", "normalize", "max_seq_length", "batch_size",
                      "trust_remote_code"):
            with self.subTest(field=field):
                self.assertNotIn(f'request.get("{field}"', self.source())


if __name__ == "__main__":
    unittest.main()
