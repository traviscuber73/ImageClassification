"""Local image classification UI for 01_PyTorch_Intel_GPU.ipynb.

Uses the existing torch, torchvision and Pillow installation. No web framework
or package installation is needed. Only loopback connections are accepted.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import secrets
import threading
import time
import traceback
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024


class UserError(Exception):
    pass


class Classifier:
    def __init__(self, project: Path):
        self.project = project
        self.models_dir = (project / "models").resolve()
        self.operation = threading.Lock()
        self.state_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.model = None
        self.info = {
            "state": "loading", "message": "저장된 모델을 준비하고 있어요.",
            "model_id": "", "device": "xpu", "device_name": "Intel GPU",
            "class_names": [], "epochs": 0, "image_size": [],
            "gpu_available": None,
        }

    def update(self, **values):
        with self.state_lock:
            self.info.update(values)

    def available_models(self):
        if not self.models_dir.is_dir():
            return []
        files = sorted(self.models_dir.glob("**/imageclassifier.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"id": p.relative_to(self.models_dir).as_posix(),
                 "label": p.parent.name} for p in files
                if p.is_file() and p.resolve().is_relative_to(self.models_dir)]

    def status(self):
        with self.state_lock:
            result = dict(self.info)
        result["models"] = self.available_models()
        result["token"] = self.token
        return result

    def load(self, model_id="", device="xpu"):
        if device not in ("xpu", "cpu"):
            raise UserError("사용할 장치를 다시 선택해 주세요.")
        if not self.operation.acquire(blocking=False):
            raise UserError("현재 작업을 마친 뒤 다시 시도해 주세요.")
        self.update(state="loading", message="PyTorch와 저장된 모델을 불러오는 중이에요.")
        threading.Thread(target=self._load, args=(model_id, device), daemon=True).start()

    def _load(self, model_id, device):
        try:
            print("[모델] PyTorch 불러오는 중...", flush=True)
            import torch
            from torch import nn
            from torchvision import transforms

            available = self.available_models()
            if not available:
                raise UserError("저장된 PyTorch 모델이 없어요. 노트북 7단계에서 학습을 완료하고 다시 불러와 주세요.")
            selected = model_id or available[0]["id"]
            if selected not in {item["id"] for item in available}:
                raise UserError("선택한 모델 파일을 찾을 수 없어요. 목록을 새로고침해 주세요.")
            path = (self.models_dir / selected).resolve()
            gpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
            self.update(gpu_available=gpu_available)
            if device == "xpu" and not gpu_available:
                raise UserError("이 Python 환경에서 Intel GPU를 사용할 수 없어요. 아래에서 CPU를 선택하거나 .venv-torch-xpu 환경을 확인해 주세요.")

            print(f"[모델] 읽는 파일: {path}", flush=True)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "SmallCNN-v1":
                raise UserError("이 파일은 현재 노트북의 SmallCNN-v1 저장 형식이 아니에요.")
            names = checkpoint.get("class_names")
            size = checkpoint.get("image_size")
            if not isinstance(names, list) or len(names) != 2 or not all(isinstance(n, str) for n in names):
                raise UserError("모델에 저장된 두 클래스 이름을 확인할 수 없어요.")
            if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(isinstance(n, int) and 1 <= n <= 1024 for n in size):
                raise UserError("모델에 저장된 이미지 크기가 올바르지 않아요.")

            # Same architecture and layer names as notebook step 6.
            class SmallCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.AdaptiveAvgPool2d((4, 4)),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 1),
                    )

                def forward(self, images):
                    return self.classifier(self.features(images)).squeeze(1)

            network = SmallCNN()
            network.load_state_dict(checkpoint["model_state_dict"], strict=True)
            target = torch.device("xpu:0" if device == "xpu" else "cpu")
            self.update(message="모델을 선택한 장치로 옮기는 중이에요.")
            network = network.to(target).eval()
            if device == "xpu":
                torch.xpu.synchronize()
            self.torch, self.device, self.model = torch, target, network
            self.transform = transforms.Compose([transforms.Resize(tuple(size)), transforms.ToTensor()])
            self.names = list(names)
            self.update(
                state="ready", message="사진을 선택하고 분류를 시작해 보세요.",
                model_id=selected, device=device,
                device_name=torch.xpu.get_device_name(0) if device == "xpu" else "CPU",
                class_names=list(names), epochs=int(checkpoint.get("epochs_completed", 0)),
                image_size=list(size),
            )
            print(f"[준비 완료] {selected} | {target}", flush=True)
        except Exception as exc:
            self.model = None
            self.update(state="error", message=str(exc))
            traceback.print_exc()
        finally:
            self.operation.release()

    def predict(self, payload):
        if not self.operation.acquire(blocking=False):
            raise UserError("모델이 다른 작업을 처리하고 있어요. 잠시 후 다시 시도해 주세요.")
        ready = False
        try:
            if self.model is None or self.status()["state"] != "ready":
                raise UserError("모델 준비가 끝난 뒤 분류해 주세요.")
            ready = True
            self.update(state="busy", message="사진을 분류하고 있어요.")
            from PIL import Image, ImageOps, UnidentifiedImageError

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(payload)) as source:
                        if source.format not in ("JPEG", "PNG", "WEBP", "BMP"):
                            raise UserError("JPG, PNG, WEBP, BMP 사진을 선택해 주세요.")
                        if source.width * source.height > 32_000_000:
                            raise UserError("사진이 너무 커요. 3,200만 화소 이하로 줄여 주세요.")
                        # Honor phone-camera rotation; use the notebook's RGB/Resize/ToTensor pipeline.
                        rgb = ImageOps.exif_transpose(source).convert("RGB")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise UserError("사진을 읽을 수 없어요. 다른 JPG 또는 PNG 파일을 선택해 주세요.") from exc

            started = time.perf_counter()
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                probability = self.torch.sigmoid(self.model(tensor)).item()
            if not math.isfinite(probability):
                raise UserError("모델이 정상적인 점수를 계산하지 못했어요. 저장된 모델을 확인해 주세요.")
            elapsed = time.perf_counter() - started
            result = {
                "predicted_name": self.names[int(probability > 0.5)],
                "probabilities": [{"name": self.names[0], "value": 1 - probability},
                                  {"name": self.names[1], "value": probability}],
                "seconds": elapsed, "device": str(self.device),
                "model_id": self.status()["model_id"],
            }
            print(f"[분류] {result['predicted_name']} | {elapsed:.2f}초", flush=True)
            return result
        finally:
            if ready:
                self.update(state="ready", message="다른 사진도 분류할 수 있어요.")
            self.operation.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "ImageLab/1.0"

    def log_message(self, fmt, *args):
        # Status polling does not flood the terminal.
        if self.path != "/api/status":
            super().log_message(fmt, *args)

    def reply(self, code, payload, mime="application/json; charset=utf-8"):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8") if isinstance(payload, dict) else payload
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        return self.headers.get("Host", "") in self.server.allowed_hosts

    def do_GET(self):
        if not self.valid_host():
            self.reply(403, {"error": "이 페이지의 로컬 주소로 접속해 주세요."})
            return
        path = urlsplit(self.path).path
        if path == "/api/status":
            self.reply(200, self.server.classifier.status())
            return
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/index.html": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/favicon.svg": ("favicon.svg", "image/svg+xml")}
        if path not in files:
            self.reply(404, {"error": "페이지를 찾을 수 없어요."})
            return
        name, mime = files[path]
        self.reply(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.valid_host() or (origin and origin not in self.server.allowed_origins):
            self.reply(403, {"error": "로컬 페이지에서 다시 시도해 주세요."})
            return
        if not secrets.compare_digest(self.headers.get("X-App-Token", ""), self.server.classifier.token):
            self.reply(403, {"error": "페이지를 새로고침해 주세요."})
            return
        path = urlsplit(self.path).path
        if path not in ("/api/predict", "/api/model"):
            self.reply(404, {"error": "요청한 기능을 찾을 수 없어요."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = MAX_UPLOAD if path == "/api/predict" else 4096
            if not 0 < length <= limit:
                self.reply(413, {"error": "20MB 이하의 사진을 선택해 주세요."})
                return
            self.connection.settimeout(30)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise UserError("사진 전송이 끝나지 않았어요. 다시 선택해 주세요.")
            if path == "/api/predict":
                result = self.server.classifier.predict(payload)
                self.reply(200, result)
            else:
                options = json.loads(payload)
                if not isinstance(options, dict):
                    raise UserError("모델 설정이 올바르지 않아요.")
                self.server.classifier.load(options.get("model_id", ""), options.get("device", "xpu"))
                self.reply(202, {"ok": True})
        except (UserError, ValueError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self.reply(500, {"error": f"분류 중 오류가 발생했어요: {exc}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    engine = Classifier(project)
    server = None
    for port in range(args.port, min(args.port + 10, 65536)):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError as exc:
            if exc.errno not in (98, 48, 10048) and getattr(exc, "winerror", None) != 10048:
                raise
    if server is None:
        raise RuntimeError("사용할 포트가 없어요. 이전 웹 실행 창을 종료해 주세요.")
    server.daemon_threads = True
    server.classifier = engine
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    server.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    server.allowed_origins = {f"http://{host}" for host in server.allowed_hosts}
    print(f"\nImage Lab: {url}\nProject: {project}\n종료: 이 창에서 Ctrl+C\n", flush=True)
    engine.load()
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n웹 실행을 종료했습니다.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
