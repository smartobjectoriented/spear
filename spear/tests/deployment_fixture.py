"""One fictional, complete deployment for the tests that need one.

TrainingExecutionConfiguration no longer defaults its host, its remote root or
its inference-service paths. It used to default them to one institute's
machine, which meant every test that constructed a bare configuration was
quietly asserting against that deployment -- and printing its hostname into
failure output.

The values here name nothing real and cannot be mistaken for a deployment:
`.example` is reserved by RFC 2606, `/srv/training` is a generic root, and the
GPU UUID is zeroes. A test that needs a different value overrides it; a test
about a MISSING value leaves it out.
"""

from training_launcher import TrainingExecutionConfiguration

HOST = "gpu-host.example"
REMOTE_ROOT = "/srv/training"
INFERENCE_BINARY = "/srv/inference/bin/llama-server"
INFERENCE_MODEL = "/srv/inference/models/model-Q8_0.gguf"
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000000"
REVISION = "a" * 40


def deployment(**changes) -> TrainingExecutionConfiguration:
    """A configuration that passes validate(), with `changes` applied."""
    value = dict(host=HOST, remote_root=REMOTE_ROOT,
                 inference_service_binary=INFERENCE_BINARY,
                 inference_service_model=INFERENCE_MODEL,
                 source_model_revision=REVISION)
    value.update(changes)

    return TrainingExecutionConfiguration(**value)
