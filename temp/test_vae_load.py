# Test loading user's workflow models (MiniMax VAE etc) - do they register successfully?
import json, time, urllib.request

BASE = "http://127.0.0.1:8199"

def post(path, data):
    req = urllib.request.Request(BASE + path, data=json.dumps(data, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"HTTP_ERROR": e.code, "body": e.read().decode()[:500]}

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())

def run_loader(slot_name):
    prompt = {
        "1": {"class_type": "ZouyuModelLoader", "inputs": {
            "model_0_type": "视频VAE",
            "model_0_name": slot_name,
            "low_vram_mode": True, "language": "中文",
        }},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "zouyu_test"}},
    }
    r = post("/prompt", {"prompt": prompt})
    if "prompt_id" not in r:
        return "SUBMIT_ERR: " + json.dumps(r, ensure_ascii=False)[:300]
    for _ in range(40):
        time.sleep(2)
        st = get("/queue")
        if not st["queue_running"] and not st["queue_pending"]:
            break
    s = get("/zouyu_model_loader/status")
    m = next((x for x in s["models"] if x["kind"] == "slot0"), None)
    if m:
        return "OK state=" + m["state"] + " name=" + m["name"]
    events = s["events"][-3:]
    return "NOT_REGISTERED events=" + " | ".join(events)

print("=== MiniMax video VAE (user workflow) ===")
print(run_loader("minimax_h3_video_vae_fp16.safetensors"))
print("=== MiniMax audio VAE ===")
print(run_loader("minimax_h3_audio_vae_fp32.safetensors"))
print("=== LTX VAE (control) ===")
print(run_loader("LTX23_video_vae_bf16.safetensors"))
