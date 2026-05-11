import sys
import tempfile
import threading
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))

import app
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
    ("NVIDIA GeForce GTX 1050 Ti", "Pascal", "cpu"),
    ("NVIDIA TITAN Xp", "Pascal", "cpu"),
    ("Tesla P100", "Pascal", "cpu"),
    ("Tesla V100", "Volta", "cpu"),
    ("NVIDIA GeForce GTX 980", "Maxwell", "cpu"),
    ("NVIDIA GeForce GTX TITAN X", "Maxwell", "cpu"),
    ("NVIDIA GeForce GTX 750 Ti", "Maxwell", "cpu"),
    ("NVIDIA GeForce GTX TITAN Black", "Kepler", "cpu"),
    ("Tesla K80", "Kepler", "cpu"),
]


for gpu_name, expected_architecture, expected_accelerator in CASES:
    architecture = runtime_installer.infer_gpu_architecture(gpu_name)
    accelerator, _ = runtime_installer.choose_torch_index("", architecture)
    assert architecture == expected_architecture, (gpu_name, architecture, expected_architecture)
    assert accelerator == expected_accelerator, (gpu_name, accelerator, expected_accelerator)

assert runtime_installer.choose_torch_index("13.0", "Kepler")[0] == "cpu"
assert runtime_installer.choose_torch_index("13.0", "Pascal")[0] == "cpu"
assert runtime_installer.choose_torch_index("13.0", "Blackwell")[0] == "cu128"
assert runtime_installer.choose_torch_index("13.0", "Pascal", "legacy-cuda")[0] == "cu118"
assert runtime_installer.choose_torch_index("13.0", "Pascal", "stable-cpu")[0] == "cpu"
assert runtime_installer.choose_torch_index("13.0", "Kepler", "legacy-cuda", "Tesla K80", "3.7")[0] == "cpu"
assert runtime_installer.choose_torch_index("13.0", "Maxwell", "legacy-cuda", "NVIDIA GeForce GTX 980", "5.2")[0] == "cu118"
assert runtime_installer.choose_torch_index("13.0", "", "auto", "NVIDIA GeForce GTX 1050 Ti", "6.1")[0] == "cpu"
assert runtime_installer.normalize_accelerator_mode("legacy_cuda") == "legacy-cuda"
assert runtime_installer.normalize_accelerator_mode("broken") == "auto"
assert runtime_installer.is_legacy_cuda_architecture("Pascal") is True
assert runtime_installer.is_legacy_cuda_architecture("Blackwell") is False

LEGACY_COMPAT_CASES = [
    ("NVIDIA GeForce GTX 580", "Fermi", "2.0", False, "unsupported"),
    ("Tesla K40", "Kepler", "3.5", False, "unsupported"),
    ("Tesla K80", "Kepler", "3.7", False, "unsupported"),
    ("NVIDIA GeForce GTX 750 Ti", "Maxwell", "5.0", True, "high-risk"),
    ("NVIDIA GeForce GTX 980", "Maxwell", "5.2", True, "high-risk"),
    ("NVIDIA GeForce GTX TITAN X", "Maxwell", "5.2", True, "high-risk"),
    ("NVIDIA GeForce GTX 1050 Ti", "Pascal", "6.1", True, "legacy-compatible"),
    ("NVIDIA TITAN Xp", "Pascal", "6.1", True, "legacy-compatible"),
    ("Tesla P100", "Pascal", "6.0", True, "legacy-compatible"),
    ("Tesla V100", "Volta", "7.0", True, "legacy-compatible"),
    ("NVIDIA GeForce GTX 1660 Ti", "Turing", "7.5", False, "modern"),
]
for gpu_name, architecture, capability, expected_available, expected_status in LEGACY_COMPAT_CASES:
    info = runtime_installer.legacy_cuda_compatibility(gpu_name, architecture, capability)
    assert info["legacy_cuda_available"] is expected_available, (gpu_name, info)
    assert info["status"] == expected_status, (gpu_name, info)

with tempfile.TemporaryDirectory() as temp_root:
    bom_json = Path(temp_root) / "config.json"
    bom_json.write_text('{"ok": true}', encoding="utf-8-sig")
    assert backend.load_json(str(bom_json)) == {"ok": True}


captured: list[tuple[str, dict]] = []
original_emit = backend.emit
original_run_runtime_preflight_subprocess = backend.run_runtime_preflight_subprocess
original_detect_nvidia_environment = backend.runtime_installer.detect_nvidia_environment
original_choose_torch_index = backend.runtime_installer.choose_torch_index
original_build_accelerator_summary = backend.runtime_installer.build_accelerator_summary

try:
    backend.emit = lambda tag, payload: captured.append((tag, payload))
    backend.run_runtime_preflight_subprocess = lambda include_extra_index=True: (
        backend.runtime_preflight.build_broken_runtime_report("No module named 'torch._inductor.runtime'")
    )
    backend.runtime_installer.detect_nvidia_environment = lambda: {
        "available": True,
        "gpu_name": "NVIDIA GeForce GTX 1050 Ti",
        "gpu_architecture": "Pascal",
        "compute_capability": "6.1",
        "driver_version": "999.99",
        "cuda_version": "13.0",
    }
    backend.runtime_installer.choose_torch_index = lambda cuda, arch, mode="auto", name="", cap="": (
        "cu118" if mode == "legacy-cuda" else "cpu",
        "https://download.pytorch.org/whl/cu118" if mode == "legacy-cuda" else "",
    )
    backend.runtime_installer.build_accelerator_summary = lambda accelerator, gpu: {
        "accelerator_label": f"NVIDIA 显卡版（{accelerator.upper()}）" if accelerator != "cpu" else "CPU 版",
        "hardware_label": "已检测到 NVIDIA 显卡：NVIDIA GeForce GTX 1050 Ti，Pascal，CUDA 13.0",
    }

    assert backend.run_check(SimpleNamespace()) == 0
    result_payload = next(payload for tag, payload in captured if tag == "RESULT")
    assert result_payload["kind"] == "check"
    assert result_payload["runtime_backend"] == "broken"
    assert "preflight_error" in result_payload
    assert result_payload["accelerator"] == "cpu"
finally:
    backend.emit = original_emit
    backend.run_runtime_preflight_subprocess = original_run_runtime_preflight_subprocess
    backend.runtime_installer.detect_nvidia_environment = original_detect_nvidia_environment
    backend.runtime_installer.choose_torch_index = original_choose_torch_index
    backend.runtime_installer.build_accelerator_summary = original_build_accelerator_summary

original_plan_detect_nvidia_environment = runtime_installer.detect_nvidia_environment
original_rank_candidate_urls = runtime_installer.rank_candidate_urls
original_detect_installed_torch_accelerator = runtime_installer.detect_installed_torch_accelerator
original_detect_pip_bootstrap_strategy = runtime_installer.detect_pip_bootstrap_strategy

try:
    runtime_installer.detect_nvidia_environment = lambda: {
        "available": True,
        "gpu_name": "NVIDIA GeForce GTX 1050 Ti",
        "gpu_architecture": "Pascal",
        "compute_capability": "6.1",
        "driver_version": "999.99",
        "cuda_version": "13.0",
    }
    runtime_installer.rank_candidate_urls = lambda candidates, logger=None: list(candidates)[:1]
    runtime_installer.detect_installed_torch_accelerator = lambda: ""
    runtime_installer.detect_pip_bootstrap_strategy = lambda: "pip-ready"

    plan_cpu = runtime_installer.build_install_plan(lambda _message: None, accelerator_mode="stable-cpu")
    assert plan_cpu["accelerator"] == "cpu"
    assert plan_cpu["accelerator_mode"] == "stable-cpu"

    plan_legacy = runtime_installer.build_install_plan(lambda _message: None, accelerator_mode="legacy-cuda")
    assert plan_legacy["accelerator"] == "cu118"
    assert plan_legacy["accelerator_mode"] == "legacy-cuda"
    assert plan_legacy["embedded_python_version"] == runtime_installer.LEGACY_CUDA_EMBEDDED_PYTHON_VERSION
    assert "cu118" in plan_legacy["torch_index"]
    assert "torch==2.7.1+cu118" in plan_legacy["torch_packages"]
    legacy_torch_command = plan_legacy["commands"][1][0]
    assert legacy_torch_command == ["__DOWNLOAD_TORCH_WHEELS__"]
    legacy_wheels = plan_legacy["torch_wheel_downloads"]
    assert len(legacy_wheels) >= 3
    expected_filenames = {
        runtime_installer.build_torch_wheel_filename("torch==2.7.1+cu118"),
        runtime_installer.build_torch_wheel_filename("torchvision==0.22.1+cu118"),
        runtime_installer.build_torch_wheel_filename("torchaudio==2.7.1+cu118"),
    }
    assert expected_filenames.issubset({str(item["filename"]) for item in legacy_wheels})
    for item in legacy_wheels[:3]:
        urls = item.get("urls") or []
        assert urls
        assert "%2Bcu118" in str(urls[0])
        assert "+cu118" not in str(urls[0])
    if len(legacy_wheels) > 3:
        assert any(item.get("category") == "torch-runtime" for item in legacy_wheels)
    local_install = runtime_installer.install_torch_wheels(
        [Path(str(item["target_path"])) for item in legacy_wheels],
        force_reinstall=False,
        dependency_index=str(plan_legacy["pip_dependency_index"]),
        dependency_indexes=list(plan_legacy["pip_dependency_indexes"]),
    )
    assert local_install[:6] == [sys.executable, "-m", "pip", "--isolated", "install", "--upgrade"]
    if "--no-index" in local_install:
        assert "--find-links" in local_install
        assert "-i" not in local_install
    else:
        assert "-i" in local_install
        assert str(plan_legacy["pip_dependency_index"]) in local_install
    assert plan_legacy["pip_dependency_indexes"] == [plan_legacy["pip_dependency_index"]]
    assert plan_legacy["command_extra_index_flags"] == [True, False, True]

    assert runtime_installer.choose_embedded_python_url(
        runtime_installer.LEGACY_CUDA_EMBEDDED_PYTHON_VERSION
    ).endswith("/python-3.12.10-embed-amd64.zip")
finally:
    runtime_installer.detect_nvidia_environment = original_plan_detect_nvidia_environment
    runtime_installer.rank_candidate_urls = original_rank_candidate_urls
    runtime_installer.detect_installed_torch_accelerator = original_detect_installed_torch_accelerator
    runtime_installer.detect_pip_bootstrap_strategy = original_detect_pip_bootstrap_strategy

backend.validate_configured_runtime({"accelerator": "cpu"}, {"runtime_backend": "cpu"})
backend.validate_configured_runtime({"accelerator": "cu118"}, {"runtime_backend": "nvidia"})
try:
    backend.validate_configured_runtime(
        {"accelerator": "cu118", "accelerator_label": "NVIDIA 显卡版（CU118）"},
        {"runtime_backend": "cpu", "runtime_backend_label": "褰撳墠杩愯鐜浣跨敤 CPU", "torch_version": "2.11.0+cpu"},
    )
except RuntimeError as exc:
    assert "显卡 CUDA 方案" in str(exc)
    assert "2.11.0+cpu" in str(exc)
else:
    raise AssertionError("CUDA plan must fail when final runtime is CPU")

process_env = runtime_installer._process_env(include_extra_index=False)
assert process_env["PIP_NO_INPUT"] == "1"
assert "PIP_EXTRA_INDEX_URL" not in process_env

with tempfile.TemporaryDirectory() as temp_root:
    temp_dir = Path(temp_root)
    payload = bytes((index % 251 for index in range(1024 * 1024)))
    source_file = temp_dir / "large.bin"
    source_file.write_bytes(payload)
    output_file = temp_dir / "downloaded.bin"

    class RangeHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args, **kwargs: RangeHandler(*args, directory=str(temp_dir), **kwargs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_min_bytes = runtime_installer.SEGMENTED_DOWNLOAD_MIN_BYTES
    try:
        runtime_installer.SEGMENTED_DOWNLOAD_MIN_BYTES = 128 * 1024
        url = f"http://127.0.0.1:{server.server_address[1]}/large.bin"
        runtime_installer.download_file(url, output_file, lambda _message: None, prefer_segmented=True)
        assert output_file.read_bytes() == payload
    finally:
        runtime_installer.SEGMENTED_DOWNLOAD_MIN_BYTES = original_min_bytes
        server.shutdown()
        server.server_close()

with tempfile.TemporaryDirectory() as temp_root:
    temp_dir = Path(temp_root)
    payload = bytes((index % 251 for index in range(256 * 1024)))
    source_file = temp_dir / "resume.bin"
    source_file.write_bytes(payload)
    output_file = temp_dir / "resume_downloaded.bin"
    part_file = output_file.with_name(output_file.name + ".part")
    part_file.write_bytes(payload[:64 * 1024])

    class ResumeHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args, **kwargs: ResumeHandler(*args, directory=str(temp_dir), **kwargs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/resume.bin"
        runtime_installer.download_file(url, output_file, lambda _message: None, prefer_segmented=False)
        assert output_file.read_bytes() == payload
        assert not part_file.exists()
    finally:
        server.shutdown()
        server.server_close()

with tempfile.TemporaryDirectory() as temp_root:
    temp_dir = Path(temp_root)
    payload = bytes((index % 251 for index in range(768 * 1024)))
    source_file = temp_dir / "segmented_resume.bin"
    source_file.write_bytes(payload)
    output_file = temp_dir / "segmented_resume_downloaded.bin"
    part_dir = output_file.with_name(output_file.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    original_min_bytes = runtime_installer.SEGMENTED_DOWNLOAD_MIN_BYTES
    original_connections = runtime_installer.DEFAULT_SEGMENTED_DOWNLOAD_CONNECTIONS

    class SegmentedResumeHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args, **kwargs: SegmentedResumeHandler(*args, directory=str(temp_dir), **kwargs))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        runtime_installer.SEGMENTED_DOWNLOAD_MIN_BYTES = 128 * 1024
        runtime_installer.DEFAULT_SEGMENTED_DOWNLOAD_CONNECTIONS = 4
        segment_count = 4
        segment_size = (len(payload) + segment_count - 1) // segment_count
        for index in range(segment_count):
            start = index * segment_size
            if start >= len(payload):
                break
            end = min(len(payload), start + segment_size)
            partial_end = start + max(1, (end - start) // 2)
            (part_dir / f"{index:03d}.part").write_bytes(payload[start:partial_end])
        url = f"http://127.0.0.1:{server.server_address[1]}/segmented_resume.bin"
        runtime_installer.download_file(url, output_file, lambda _message: None, prefer_segmented=True)
        assert output_file.read_bytes() == payload
        assert not output_file.with_name(output_file.name + ".part").exists()
        assert not part_dir.exists()
    finally:
        runtime_installer.SEGMENTED_DOWNLOAD_MIN_BYTES = original_min_bytes
        runtime_installer.DEFAULT_SEGMENTED_DOWNLOAD_CONNECTIONS = original_connections
        server.shutdown()
        server.server_close()


with tempfile.TemporaryDirectory() as temp_root:
    temp_dir = Path(temp_root)
    output_file = temp_dir / "cached.whl"
    part_file = output_file.with_name(output_file.name + ".part")
    with zipfile.ZipFile(part_file, "w") as archive:
        archive.writestr("ok.txt", "ok")
    recovered_path, recovered_source = runtime_installer.download_file(
        "http://127.0.0.1:9/cached.whl",
        output_file,
        lambda _message: None,
    )
    assert recovered_path == output_file
    assert recovered_source == "cache"
    assert output_file.exists()
    assert not part_file.exists()

with tempfile.TemporaryDirectory() as temp_root:
    temp_dir = Path(temp_root)
    output_file = temp_dir / "switch_source.whl"
    bad_file = temp_dir / "bad.whl"
    good_file = temp_dir / "good.whl"
    bad_file.write_bytes(b"not a wheel archive")
    with zipfile.ZipFile(good_file, "w") as archive:
        archive.writestr("ok.txt", "ok")

    class SwitchSourceHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        lambda *args, **kwargs: SwitchSourceHandler(*args, directory=str(temp_dir), **kwargs),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        recovered_path, recovered_source = runtime_installer.download_file(
            [f"{base_url}/bad.whl", f"{base_url}/good.whl"],
            output_file,
            lambda _message: None,
        )
        assert recovered_path == output_file
        assert recovered_source.endswith("/good.whl")
        assert zipfile.is_zipfile(output_file)
        assert not output_file.with_name(output_file.name + ".part").exists()
        assert not output_file.with_name(output_file.name + ".parts").exists()
    finally:
        server.shutdown()
        server.server_close()

same_host_urls = [
    "https://files.pythonhosted.org/packages/a/b/demo-1.0.0-py3-none-any.whl",
    "https://mirrors.aliyun.com/pypi/packages/a/b/demo-1.0.0-py3-none-any.whl",
    "https://mirror.sjtu.edu.cn/pypi/packages/a/b/demo-1.0.0-py3-none-any.whl",
]
ordered_same_host = runtime_installer._sort_urls_by_index_order(
    same_host_urls,
    [
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://mirror.sjtu.edu.cn/pypi/web/simple/",
    ],
)
assert ordered_same_host[0].startswith("https://mirrors.aliyun.com/")
assert ordered_same_host[1].startswith("https://mirror.sjtu.edu.cn/")
assert ordered_same_host[-1].startswith("https://files.pythonhosted.org/")


class DummyStringVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class DummyApp:
    def __init__(self) -> None:
        self.python_var = DummyStringVar()
        self.started_command: list[str] | None = None
        self.thread_accelerator_mode: str | None = None

    def _is_builtin_runtime_target(self, _path: str) -> bool:
        return True

    def _start_process(self, command: list[str], **_kwargs: object) -> None:
        self.started_command = command

    def _start_bootstrap_runtime_and_configure_thread(self, accelerator_mode: str) -> None:
        self.thread_accelerator_mode = accelerator_mode


original_is_frozen_app = app.is_frozen_app
try:
    app.is_frozen_app = lambda: True
    dummy = DummyApp()
    app.App._start_configure_process(dummy, sys.executable, "legacy-cuda")
    assert dummy.started_command is None
    assert dummy.thread_accelerator_mode == "legacy-cuda"
    assert dummy.python_var.value == str(app.BUNDLED_RUNTIME_PYTHON)

    dummy = DummyApp()
    app.App._start_configure_process(dummy, sys.executable, "stable-cpu")
    assert dummy.started_command is not None
    assert "configure-env" in dummy.started_command
    assert "bootstrap-runtime-and-configure" not in dummy.started_command
finally:
    app.is_frozen_app = original_is_frozen_app

print("RUNTIME_DETECTION_SELFTEST_OK")
