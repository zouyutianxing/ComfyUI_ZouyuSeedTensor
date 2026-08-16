# Verify: (1) register_config clears stale registered model, (2) unload in CPU-cache mode now unloads to RAM
import json, time, urllib.request

BASE = "http://127.0.0.1:8199"

def post(path, data):
    req = urllib.request.Request(BASE + path, data=json.dumps(data, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())

def run_loader(slot_name):
    prompt = {
        "1": {"class_type": "ZouyuModelLoader", "inputs": {
            "model_0_type": "视频VAE",
            "model_0_name": slot_name,
            "low_vram_mode": False, "language": "中文",
        }},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "zouyu_test"}},
    }
    post("/prompt", {"prompt": prompt})
    for _ in range(40):
        time.sleep(2)
        st = get("/queue")
        if not st["queue_running"] and not st["queue_pending"]:
            return

def state():
    s = get("/zouyu_model_loader/status")
    m = next((x for x in s["models"] if x["kind"] == "slot0"), None)
    return (m["state"] if m else "none"), (m["name"] if m else "")

# 1) run loader with LTX -> registered slot0=LTX
print("=== 1) load LTX VAE ===")
run_loader("LTX23_video_vae_bf16.safetensors")
print("  registered:", state())
# 2) user re-configures slot0 = minimax via register_config -> stale LTX cleared
print("=== 2) re-configure slot0 = minimax (register_config) ===")
post("/zouyu_model_loader/register_config", {"slots": [{"slot": 0, "tkey": "vae", "folder": "", "name": "minimax_h3_video_vae_fp16.safetensors"}]})
s = get("/zouyu_model_loader/status")
m = next((x for x in s["models"] if x["kind"] == "slot0"), None)
print("  status slot0 name:", m["name"] if m else "none", "| state:", m["state"] if m else "none")
print("  PASS if name=minimax (stale LTX cleared)")

# 3) load minimax -> registered, then unload in CPU-cache mode -> should unload to RAM
print("=== 3) load minimax, then unload (CPU-cache) ===")
run_loader("minimax_h3_video_vae_fp16.safetensors")
print("  after load:", state())
r = post("/zouyu_model_loader/slot_action", {"kind": "slot0", "action": "unload"})
print("  unload msg:", r.get("message"))
print("  after unload:", state())
print("  PASS if msg contains 卸载至CPU内存 (official standard)")
