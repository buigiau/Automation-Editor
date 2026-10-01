"""Exercise real selection/recovery boundaries without requiring GPU hardware."""
from types import SimpleNamespace as NS
import os
import sys

import numpy as np
import pytest

from autoedit.acceleration import device_settings
from autoedit.audio import source_speech as speech
from autoedit.video import embeddings


@pytest.fixture
def whisper_factory(monkeypatch):
    calls = []
    failures = {}

    class FakeWhisper:
        def __init__(self, path, **kwargs):
            self.device = kwargs["device"]
            calls.append(kwargs)
            if failures.get(self.device):
                raise failures[self.device]

        def transcribe(self, samples, **kwargs):
            return iter([self.device]), NS()

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeWhisper)
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(speech, "prepare_nvidia_runtime", lambda: None)
    monkeypatch.setattr(speech, "_MODELS", {})
    return calls, failures, FakeWhisper


def test_whisper_auto_selects_gpu_and_cpu_override(whisper_factory):
    calls, _, _ = whisper_factory
    gpu = speech._WhisperRuntime("model", "auto", "auto", 0)
    cpu = speech._WhisperRuntime("model", "cpu", "auto", 0)
    assert gpu.execution["device"] == "cuda"
    assert calls[0]["compute_type"] == "int8_float16"
    assert cpu.execution["device"] == "cpu"
    assert calls[1]["compute_type"] == "int8"


def test_whisper_no_gpu_auto_recovers_but_explicit_cuda_fails(whisper_factory, monkeypatch):
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", lambda: 0)
    runtime = speech._WhisperRuntime("model", "auto", "auto", 0)
    assert runtime.device == "cpu" and runtime.fallback_reason
    with pytest.raises(RuntimeError, match="CUDA device"):
        speech._WhisperRuntime("model", "cuda", "auto", 0)


def test_whisper_missing_dll_auto_recovers_and_reports(whisper_factory):
    calls, failures, _ = whisper_factory
    failures["cuda"] = RuntimeError("cublas64_12.dll is not found or cannot be loaded")
    logs = []
    runtime = speech._WhisperRuntime("model", "auto", "float16", 0, logs.append)
    assert [c["device"] for c in calls] == ["cuda", "cpu"]
    assert runtime.compute_type == "int8"
    assert "CPU fallback" in logs[-1]
    with pytest.raises(RuntimeError, match="cublas"):
        speech._WhisperRuntime("model", "cuda", "auto", 0)


def test_whisper_cpu_recovery_preserves_supported_explicit_precision(whisper_factory):
    _, failures, _ = whisper_factory
    failures["cuda"] = RuntimeError("CUDA driver unavailable")
    runtime = speech._WhisperRuntime("model", "auto", "float32", 0)
    assert runtime.device == "cpu" and runtime.compute_type == "float32"


def test_whisper_lazy_gpu_failure_discards_partial_chunk_and_stays_on_cpu(whisper_factory):
    calls, _, factory = whisper_factory
    runtime = speech._WhisperRuntime("model", "auto", "auto", 0)

    def broken_segments():
        yield "partial GPU segment"
        raise RuntimeError("CUDA out of memory")

    runtime.model.transcribe = lambda *a, **kw: (broken_segments(), NS())
    segments, _ = runtime.transcribe(np.zeros(32000))
    assert list(segments) == ["cpu"]
    assert runtime.execution["device"] == "cpu"
    assert list(runtime.transcribe(np.zeros(32000))[0]) == ["cpu"]
    assert len(calls) == 2


def test_whisper_explicit_cuda_does_not_hide_deferred_error(whisper_factory):
    runtime = speech._WhisperRuntime("model", "cuda", "auto", 0)

    def broken_segments():
        yield "partial"
        raise RuntimeError("CUDA failure")

    runtime.model.transcribe = lambda *a, **kw: (broken_segments(), NS())
    with pytest.raises(RuntimeError, match="CUDA failure"):
        runtime.transcribe(np.zeros(16000))
    assert len(whisper_factory[0]) == 1


@pytest.mark.parametrize("device", ["auto", "cuda", "cpu"])
def test_whisper_invalid_model_is_not_hidden_by_fallback(whisper_factory, device):
    calls, failures, _ = whisper_factory
    failures["cuda" if device != "cpu" else "cpu"] = ValueError("Invalid model file")
    with pytest.raises(ValueError, match="Invalid model"):
        speech._WhisperRuntime("model", device, "auto", 0)
    assert len(calls) == 1


def test_whisper_cache_separates_device_and_precision(tmp_path, whisper_factory):
    cpu = speech._model(str(tmp_path), device="cpu")
    gpu = speech._model(str(tmp_path), device="cuda")
    fp16 = speech._model(str(tmp_path), device="cuda", compute_type="float16")
    assert cpu is speech._model(str(tmp_path), device="cpu")
    assert cpu is not gpu and fp16 is not gpu
    assert len(whisper_factory[0]) == 3


@pytest.mark.parametrize("device,index", [("gpu", 0), ("auto", -1), ("cuda", True), ("cpu", "0")])
def test_invalid_device_settings(device, index):
    with pytest.raises(ValueError):
        device_settings(device, index)


@pytest.fixture
def onnx_factory(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test")
    calls = []
    state = {"available": True, "fail_init": None, "fail_run": None, "silent_cpu": False}

    class Session:
        def __init__(self, path, options, providers):
            self.cuda = isinstance(providers[0], tuple)
            calls.append(providers)
            if self.cuda and state["fail_init"]:
                raise state["fail_init"]
            if state["silent_cpu"]:
                self.cuda = False

        def get_inputs(self):
            return [NS(name="image", shape=[1, 3, 112, 112])]

        def get_providers(self):
            return (["CUDAExecutionProvider"] if self.cuda else []) + ["CPUExecutionProvider"]

        def disable_fallback(self):
            self.disabled = True

        def run(self, *args):
            if self.cuda and state["fail_run"]:
                raise state["fail_run"]
            return [np.array([[1, 0]], np.float32)]

    fake = NS(SessionOptions=NS, InferenceSession=Session, preload_dlls=lambda: None,
              get_available_providers=lambda:
                  (["CUDAExecutionProvider"] if state["available"] else []) + ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(embeddings, "prepare_nvidia_runtime", lambda: None)
    return model, calls, state


def test_onnx_selects_cuda_and_records_actual_provider(onnx_factory):
    path, calls, _ = onnx_factory
    model = embeddings.EmbeddingModel("live_action", {"arcface_model": str(path), "device_index": 2})
    assert calls[0][0] == ("CUDAExecutionProvider", {"device_id": 2, "use_tf32": 0,
                                                   "cudnn_conv_use_max_workspace": 0})
    assert model.execution["device"] == "cuda"
    assert model.session.disabled


@pytest.mark.parametrize("failure", ["unavailable", "init", "silent_cpu"])
def test_onnx_auto_falls_back_when_cuda_cannot_initialize(onnx_factory, failure):
    path, calls, state = onnx_factory
    if failure == "unavailable":
        state["available"] = False
    elif failure == "init":
        state["fail_init"] = RuntimeError("CUDA driver error")
    else:
        state["silent_cpu"] = True
    config = {"arcface_model": str(path)}
    model = embeddings.EmbeddingModel("live_action", config)
    assert model.execution["device"] == "cpu"
    assert model.execution["fallback_reason"]
    assert calls[-1] == ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError):
        embeddings.EmbeddingModel("live_action", {**config, "device": "cuda"})


def test_onnx_runtime_oom_retries_same_input_on_cpu(onnx_factory):
    path, calls, state = onnx_factory
    state["fail_run"] = RuntimeError("CUDA out of memory")
    model = embeddings.EmbeddingModel("live_action", {"arcface_model": str(path)})
    assert model._run(np.zeros((1, 3, 112, 112), np.float32))[0].tolist() == [[1, 0]]
    assert model.execution["device"] == "cpu"
    assert len(calls) == 2


def test_onnx_explicit_cuda_does_not_hide_runtime_error(onnx_factory):
    path, calls, state = onnx_factory
    state["fail_run"] = RuntimeError("CUDA out of memory")
    model = embeddings.EmbeddingModel("live_action", {"arcface_model": str(path), "device": "cuda"})
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        model._run(np.zeros((1, 3, 112, 112), np.float32))
    assert len(calls) == 1


def test_onnx_bad_inputs_and_invalid_models_do_not_trigger_cpu_retry(onnx_factory):
    path, calls, state = onnx_factory
    config = {"arcface_model": str(path)}
    state["fail_init"] = ValueError("Invalid model protobuf")
    with pytest.raises(ValueError):
        embeddings.EmbeddingModel("live_action", config)
    assert len(calls) == 1
    state["fail_init"] = None
    state["fail_run"] = ValueError("Invalid tensor dimensions")
    model = embeddings.EmbeddingModel("live_action", config)
    with pytest.raises(ValueError):
        model._run(np.zeros(3))
    assert len(calls) == 2


def test_onnx_cpu_override_does_not_attempt_cuda(onnx_factory):
    path, calls, state = onnx_factory
    state["fail_init"] = RuntimeError("CUDA broken")
    model = embeddings.EmbeddingModel("live_action", {"arcface_model": str(path), "device": "cpu"})
    assert calls == [["CPUExecutionProvider"]]
    assert model.execution["fallback_reason"] is None


@pytest.mark.skipif(os.environ.get("AUTOEDIT_GPU_TESTS") != "1", reason="Set AUTOEDIT_GPU_TESTS=1 for real CUDA inference")
def test_real_cuda_character_backend_preserves_vectors_on_synthetic_model(tmp_path):
    """Test runtime/preprocessing, not character recognition accuracy."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    weight = np.arange(54, dtype=np.float32).reshape(2, 3, 3, 3) / 54
    graph = helper.make_graph([
        helper.make_node("Conv", ["image", "weight"], ["conv"]),
        helper.make_node("GlobalAveragePool", ["conv"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["features"], axis=1),
    ], "cuda-smoke", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 112, 112])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(weight, "weight")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "synthetic.onnx"
    onnx.save(model, path)
    tensor = np.random.default_rng(17).random((1, 3, 112, 112), dtype=np.float32)
    cpu = embeddings.EmbeddingModel("live_action", {"arcface_model": str(path), "device": "cpu"})
    gpu = embeddings.EmbeddingModel("live_action", {"arcface_model": str(path), "device": "cuda"})
    assert gpu.execution["device"] == "cuda"
    assert "CUDAExecutionProvider" in gpu.execution["providers"]
    np.testing.assert_allclose(cpu._run(tensor)[0], gpu._run(tensor)[0], rtol=1e-4, atol=1e-4)
