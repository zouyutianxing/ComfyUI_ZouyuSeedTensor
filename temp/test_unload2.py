# Verify full unload chain: run loader (registers VAE) -> switch unload -> state change
import json, time, urllib.request

BASE = "http://127.0.0.1:8199"

def post(path, data):
    req = urllib.request.Request(BASE + path, data=json.dumps(data, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())

def run_loader(low_vram):
    prompt = {
        "1": {"class_type": "ZouyuModelLoader", "inputs": {
            "model_0_type": "视频VAE",
            "model_0_name": "LTX23_video_vae_bf16.safetensors",
            "low_vram_mode": low_vram, "language": "中文",
        }},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "zouyu_test"}},
    }
    q = post("/prompt", {"prompt": prompt})
    for _ in range(40):
        time.sleep(2)
        st = get("/queue")
        if not st["queue_running"] and not st["queue_pending"]:
            return True
    return False

def state():
    s = get("/zouyu_model_loader/status")
    m = next((x for x in s["models"] if x["kind"] == "slot0"), None)
    return (m["state"] if m else "none"), (m["zh"] if m else ""), (m["color"] if m else "")

print("=== 1) run loader with low_vram=ON (registers slot0 VAE) ===")
run_loader(True)
print("  after load:", state())
print("=== 2) switch unload signal (low_vram ON) -> should full unload ===")
r = post("/zouyu_model_loader/slot_action", {"kind": "slot0", "action": "unload"})
print("  msg:", r.get("message"))
print("  after unload:", state())

print("=== 3) run loader with low_vram=OFF (CPU cache) ===")
run_loader(False)
print("  after load:", state())
print("=== 4) switch unload signal (low_vram OFF) -> official handles ===")
r2 = post("/zouyu_model_loader/slot_action", {"kind": "slot0", "action": "unload"})
print("  msg:", r2.get("message"))
print("  after:", state())
