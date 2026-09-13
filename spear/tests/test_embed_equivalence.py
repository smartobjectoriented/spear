"""Does the new worker produce the vectors the old one produced?

The whole risk of this change is invisible. Both workers return well-formed
float32 of the right shape; if they disagree numerically, every collection
rebuilt after the cutover is subtly inconsistent with every collection built
before it, and no query reports it -- results merely get worse.

So the gate is numerical, it runs both workers for real on real weights, and
it is opt-in:

    SPEAR_TEST_EMBED_EQUIVALENCE=1 ./bin/python -m unittest \\
        tests.test_embed_equivalence

ACCEPTANCE CRITERION

Bit-exact equality, asserted -- not `allclose`. The two paths must reduce to
the same call:

    SentenceTransformer(model, device, trust_remote_code)   same weights
    .max_seq_length = min(own, 1024)                        same window
    .encode(prefix + text, batch_size, normalize=True,      same inputs,
            convert_to_numpy=True)                          same reduction

Nothing between them is allowed to differ, so any difference at all is a
defect in this change rather than floating-point noise. The looser measures
below are reported anyway, because a bit-exact assertion that starts failing
is much easier to judge with a max-absolute-difference beside it.

The one real difference is where the prefix and the cap come from -- the old
worker read the client's registry on the GPU host, the new one is told -- and
that is precisely what must not change the numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "server" / "embed"))

RUN = os.environ.get("SPEAR_TEST_EMBED_EQUIVALENCE") == "1"

#: Short, long, multilingual, punctuation-only, and one that exceeds the cap
#: so the truncation path is exercised rather than assumed.
DOCUMENTS = [
    "static int spear_probe(struct platform_device *pdev)",
    "The corpus is indexed once and queried many times.",
    "La règle normative prime sur le code et sur les commentaires.",
    "  \t\n  ",
    "#define X 1\n" * 400,
]

QUERIES = [
    "where is the interrupt handler registered",
    "quelle fonction initialise le contrôleur",
]


def measure(a, b):
    """Every number the report asks for, computed once."""
    import numpy

    a, b = numpy.asarray(a, dtype=numpy.float32), numpy.asarray(b, dtype=numpy.float32)
    delta = numpy.abs(a - b)
    cosine = float(numpy.mean([
        numpy.dot(x, y) / ((numpy.linalg.norm(x) * numpy.linalg.norm(y)) or 1.0)
        for x, y in zip(a, b)]))

    return {
        "shape_a": tuple(a.shape), "shape_b": tuple(b.shape),
        "dim": int(a.shape[1]) if a.ndim == 2 else None,
        "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
        "exact": bool(numpy.array_equal(a, b)),
        "max_abs": float(delta.max()) if delta.size else 0.0,
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "allclose_1e-6": bool(numpy.allclose(a, b, rtol=0.0, atol=1e-6)),
        "allclose_1e-5": bool(numpy.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "cosine": cosine,
    }


def report(name, stats):
    print(f"\n  [{name}]")
    for key, value in stats.items():
        print(f"      {key:14s} {value}")


@unittest.skipUnless(RUN, "set SPEAR_TEST_EMBED_EQUIVALENCE=1 for the "
                          "real-weights equivalence gate")
class TheNewWorkerReproducesTheOldOne(unittest.TestCase):
    """Both workers, both really run, on the model this deployment uses."""

    MODEL = os.environ.get("SPEAR_TEST_EMBED_MODEL", "BAAI/bge-m3")

    @classmethod
    def setUpClass(cls):
        import embedding

        cls.embedding = embedding
        cls.device = os.environ.get("SPEAR_TEST_EMBED_DEVICE", "cpu")

    def legacy(self, texts):
        """The retired worker's semantics, as it computed them.

        Reproduced through the client module exactly as spear/deploy/
        embed_worker.py did: it set the device, forced local compute, and
        called embed_documents, which read the registry for the prefix and
        capped the window. Calling the same function the same way is what
        makes this the OLD behaviour and not a restatement of the new one.
        """
        import numpy

        saved = {k: os.environ.get(k) for k in
                 ("SPEAR_EMBED_DEVICE", "SPEAR_EMBED_REMOTE")}
        os.environ["SPEAR_EMBED_DEVICE"] = self.device
        os.environ["SPEAR_EMBED_REMOTE"] = ""        # never ship it on again

        try:
            vectors = self.embedding.embed_documents(texts, self.MODEL,
                                                     batch_size=16)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)

                if value is not None:
                    os.environ[key] = value

        return numpy.asarray(vectors, dtype=numpy.float32)

    def generic(self, texts, prefix):
        """The new worker, driven over its own protocol, in its own process.

        A subprocess and not an import: running it in-process would share this
        interpreter's torch state and prove less than the deployment does.
        """
        import numpy
        import protocol

        request = protocol.encode_request(
            model=self.MODEL, texts=texts, prefix=prefix,
            max_seq_length=self.embedding.MAX_SEQ, normalize=True,
            batch_size=16, trust_remote_code=False)

        done = subprocess.run(
            [sys.executable, str(REPO / "server" / "embed" / "worker.py")],
            input=request, capture_output=True,
            env=dict(os.environ, SPEAR_EMBED_DEVICE=self.device))

        self.assertEqual(done.returncode, 0,
                         done.stdout[:400] + b" | " + done.stderr[-400:])

        header, body = protocol.decode_response(done.stdout)

        return numpy.asarray(protocol.unpack_vectors(header, body),
                             dtype=numpy.float32)

    def test_documents_are_bit_identical(self):
        prefix = self.embedding.MODELS[self.MODEL][0]
        stats = measure(self.legacy(DOCUMENTS),
                        self.generic(DOCUMENTS, prefix))
        report(f"documents · {self.MODEL} · {self.device}", stats)

        self.assertEqual(stats["shape_a"], stats["shape_b"])
        self.assertEqual(stats["dtype_a"], "float32")
        self.assertTrue(stats["exact"],
                        f"max_abs={stats['max_abs']} cosine={stats['cosine']}")

    def test_queries_are_bit_identical(self):
        """The query prefix is asymmetric for two of the four models, and
        applying it on the wrong side costs several points of recall.

        One text per request, because that is how the client embeds a query:
        `embed_query` encodes a single sentence per turn. Comparing a batch
        here against those single calls would measure the batching effect
        below instead of the change under test.
        """
        import numpy

        prefix = self.embedding.MODELS[self.MODEL][1]
        legacy = numpy.asarray(
            [self.legacy_query(text, prefix)[0] for text in QUERIES],
            dtype=numpy.float32)
        generic = numpy.concatenate(
            [self.generic([text], prefix) for text in QUERIES])

        stats = measure(legacy, generic)
        report(f"queries · {self.MODEL} · {self.device}", stats)

        self.assertTrue(stats["exact"],
                        f"max_abs={stats['max_abs']} cosine={stats['cosine']}")

    def test_batching_moves_the_last_bits_and_only_those(self):
        """A measured property of the encoder, not of this change.

        Padding a short sequence beside a long one in the same batch changes
        the final bits of the result. It is worth pinning rather than
        discovering later as a mysterious inequality: the same texts embedded
        one at a time and together are not bit-identical, and the difference
        must stay at the level of floating-point noise.

        Measured on bge-m3, max absolute difference:

            cpu,  fp32     2.1e-07     one ULP at this magnitude
            cuda, fp16     4.9e-04     ~2300x larger, and expected

        The fp16 figure is the reason the acceptance criterion above is
        bit-exact rather than `allclose`. A tolerance loose enough to cover
        the GPU's own batching noise would be 5e-04 -- which is large enough
        to hide a genuinely different encoding. Bit-exactness is available
        here precisely because both paths reduce to the same call, so there
        is no reason to accept anything less.

        It is also worth knowing operationally: changing the batch size
        changes the vectors, and on fp16 it changes them well above float32
        noise. A collection is not portable across batch sizes in the way one
        might assume -- though at cosine 0.999999 it is far below anything
        retrieval can notice.
        """
        import numpy

        prefix = self.embedding.MODELS[self.MODEL][0]
        one_at_a_time = numpy.concatenate(
            [self.generic([text], prefix) for text in DOCUMENTS])
        stats = measure(one_at_a_time, self.generic(DOCUMENTS, prefix))
        report(f"batching effect on {self.device} (informational)", stats)

        # Device-dependent, and stated per device rather than set to the
        # looser of the two: a cpu run that started drifting to 5e-04 would
        # be a real finding, and a shared bound would swallow it.
        bound = 2e-03 if self.device.startswith("cuda") else 1e-05

        self.assertLess(stats["max_abs"], bound)
        self.assertGreater(stats["cosine"], 0.9999)

    def legacy_query(self, text, prefix):
        saved = os.environ.get("SPEAR_EMBED_DEVICE")
        os.environ["SPEAR_EMBED_DEVICE"] = self.device

        try:
            return [self.embedding.embed_query(text, self.MODEL)]
        finally:
            os.environ.pop("SPEAR_EMBED_DEVICE", None)

            if saved is not None:
                os.environ["SPEAR_EMBED_DEVICE"] = saved

    def test_a_different_prefix_really_changes_the_vectors(self):
        """The gate is only meaningful if the inputs it compares can differ.

        Two encodings that agree because the prefix is ignored would pass
        every assertion above and prove nothing.
        """
        stats = measure(self.generic(DOCUMENTS, ""),
                        self.generic(DOCUMENTS, "passage: "))
        report("prefix sensitivity (must NOT be exact)", stats)

        self.assertFalse(stats["exact"])


if __name__ == "__main__":
    unittest.main()
