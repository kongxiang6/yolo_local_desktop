import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend
from vendor_backend import runtime_installer


CASES = [
    ("NVIDIA GeForce RTX 5070 Ti", "Blackwell", "cu128"),
    ("NVIDIA GeForce RTX 5090D", "Blackwell", "cu128"),
    ("NVIDIA RTX PRO 6000 Blackwell", "Blackwell", "cu128"),
    ("NVIDIA B200", "Blackwell", "cu128"),
    ("NVIDIA GeForce RTX 4090", "Ada", "cu128"),
    ("NVIDIA RTX 6000 Ada", "Ada", "cu128"),
    ("NVIDIA L4", "Ada", "cu128"),
    ("NVIDIA L40S", "Ada", "cu128"),
    ("NVIDIA GeForce RTX 3080", "Ampere", "cu128"),
    ("NVIDIA RTX A6000", "Ampere", "cu128"),
    ("NVIDIA A100", "Ampere", "cu128"),
    ("NVIDIA H100", "Hopper", "cu128"),
    ("NVIDIA GeForce RTX 2080 Ti", "Turing", "cu128"),
    ("NVIDIA GeForce GTX 1660 Ti", "Turing", "cu128"),
    ("Quadro RTX 5000", "Turing", "cu128"),
    ("NVIDIA GeForce GTX 1050 Ti", "Pascal", "cu126"),
    ("Tesla P100", "Pascal", "cu126"),
    ("Tesla V100", "Volta", "cu126"),
    ("NVIDIA GeForce GTX 980", "Maxwell", "cu126"),
    ("NVIDIA GeForce GTX 750 Ti", "Maxwell", "cu126"),
    ("Tesla K80", "Kepler", "cpu"),
]


for gpu_name, expected_architecture, expected_accelerator in CASES:
    architecture = runtime_installer.infer_gpu_architecture(gpu_name)
    accelerator, _ = runtime_installer.choose_torch_index("", architecture)
    assert architecture == expected_architecture, (gpu_name, architecture, expected_architecture)
    assert accelerator == expected_accelerator, (gpu_name, accelerator, expected_accelerator)

assert runtime_installer.choose_torch_index("13.0", "Kepler")[0] == "cpu"
assert runtime_installer.choose_torch_index("13.0", "Pascal")[0] == "cu126"
assert runtime_installer.choose_torch_index("13.0", "Blackwell")[0] == "cu128"


captured: list[tuple[str, dict]] = []
original_emit = backend.emit
original_run_runtime_preflight = backend.runtime_preflight.run_runtime_preflight
original_detect_nvidia_environment = backend.runtime_installer.detect_nvidia_environment
original_choose_torch_index = backend.runtime_installer.choose_torch_index
original_build_accelerator_summary = backend.runtime_installer.build_accelerator_summary

try:
    backend.emit = lambda tag, payload: captured.append((tag, payload))
    backend.runtime_preflight.run_runtime_preflight = lambda: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named 'torch._inductor.runtime'")
    )
    backend.runtime_installer.detect_nvidia_environment = lambda: {
        "available": True,
        "gpu_name": "NVIDIA GeForce GTX 1050 Ti",
        "gpu_architecture": "Pascal",
        "driver_version": "999.99",
        "cuda_version": "13.0",
    }
    backend.runtime_installer.choose_torch_index = lambda cuda, arch: ("cu126", "https://download.pytorch.org/whl/cu126")
    backend.runtime_installer.build_accelerator_summary = lambda accelerator, gpu: {
        "accelerator_label": "NVIDIA 显卡版（CU126）",
        "hardware_label": "已检测到 NVIDIA 显卡：NVIDIA GeForce GTX 1050 Ti，Pascal，CUDA 13.0",
    }

    assert backend.run_check(SimpleNamespace()) == 0
    result_payload = next(payload for tag, payload in captured if tag == "RESULT")
    assert result_payload["kind"] == "check"
    assert result_payload["runtime_backend"] == "broken"
    assert "preflight_error" in result_payload
    assert result_payload["accelerator"] == "cu126"
finally:
    backend.emit = original_emit
    backend.runtime_preflight.run_runtime_preflight = original_run_runtime_preflight
    backend.runtime_installer.detect_nvidia_environment = original_detect_nvidia_environment
    backend.runtime_installer.choose_torch_index = original_choose_torch_index
    backend.runtime_installer.build_accelerator_summary = original_build_accelerator_summary

print("RUNTIME_DETECTION_SELFTEST_OK")
