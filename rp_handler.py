import os
import gc
import uuid
import tempfile
import json
import requests
import subprocess
import shutil
from datetime import timedelta

import torch
from indextts.infer_v2 import IndexTTS2
from google.cloud import storage
import runpod

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

BUCKET_NAME = "runpod_bucket_testing"
CREDS_FILE_ID = "1leNukepERYsBmoKSYTbqUjGb-pQvwQlz"

log_prefix = "[DEBUG]"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IndexTeam/IndexTTS-2",
    local_dir="checkpoints"
)


def log(msg):
    print(f"{log_prefix} {msg}", flush=True)

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def fetch_gcs_json_from_drive(file_id: str, save_path="/tmp/gcs_creds.json"):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    log(f"GCS creds saved to {save_path}")

def get_gcs_client():
    creds_path = "/tmp/gcs_creds.json"
    if not os.path.exists(creds_path):
        fetch_gcs_json_from_drive(CREDS_FILE_ID, creds_path)
    return storage.Client.from_service_account_json(creds_path)

def upload_to_gcs(local_path, prefix="tts_outputs"):
    try:
        client = get_gcs_client()
        bucket = client.bucket(BUCKET_NAME)
        blob_path = f"{prefix}/{uuid.uuid4()}{os.path.splitext(local_path)[1]}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        url = blob.generate_signed_url(expiration=timedelta(hours=6))
        log(f"Uploaded to GCS: {url}")
        return url
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

def download_file(url):
    log(f"Downloading file from {url}")
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def save_temp(data, suffix=".wav"):
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    with open(temp_path, "wb") as f:
        f.write(data)
    log(f"Saved temp file: {temp_path}")
    return temp_path

# ---------------- INIT MODEL ----------------
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False
)

# ---------------- RUN infer_v2.py ONCE ----------------
try:
    log("Running indextts/infer_v2.py at startup...")
    subprocess.run(
        ["uv", "run", "indextts/infer_v2.py"],
        check=True,
        env=os.environ
    )
    log("Finished running infer_v2.py")
except subprocess.CalledProcessError as e:
    log(f"Failed to run infer_v2.py: {e}")

# ---------------- HANDLER ----------------
def handler(event):
    inp = event.get("input", {})
    task = inp.get("task")
    if not task:
        return {"status": "error", "message": "Missing 'task'"}

    uid = str(uuid.uuid4())
    out_path = f"/tmp/out_{uid}.mp3"

    try:
        if task == "tts_clone":
            spk_path = save_temp(download_file(inp["spk_url"]))
            clear_mem()
            tts.infer(spk_audio_prompt=spk_path, text=inp["text"], output_path=out_path)
            clear_mem()
            return {"status": "success", "url": upload_to_gcs(out_path)}

        if task == "tts_emotion_audio":
            spk_path = save_temp(download_file(inp["spk_url"]))
            emo_path = save_temp(download_file(inp["emo_url"]))
            clear_mem()
            tts.infer(
                spk_audio_prompt=spk_path,
                emo_audio_prompt=emo_path,
                emo_alpha=float(inp.get("emo_alpha", 1.0)),
                text=inp["text"],
                output_path=out_path
            )
            clear_mem()
            return {"status": "success", "url": upload_to_gcs(out_path)}

        if task == "tts_emotion_vector":
            spk_path = save_temp(download_file(inp["spk_url"]))
            vector = [float(x) for x in inp["emo_vector"].split(",")]
            clear_mem()
            tts.infer(
                spk_audio_prompt=spk_path,
                text=inp["text"],
                emo_vector=vector,
                use_random=inp.get("use_random", False),
                output_path=out_path
            )
            clear_mem()
            return {"status": "success", "url": upload_to_gcs(out_path)}

        if task == "tts_emotion_text_auto":
            spk_path = save_temp(download_file(inp["spk_url"]))
            clear_mem()
            tts.infer(
                spk_audio_prompt=spk_path,
                text=inp["text"],
                emo_alpha=float(inp.get("emo_alpha", 0.6)),
                use_emo_text=True,
                use_random=inp.get("use_random", False),
                output_path=out_path
            )
            clear_mem()
            return {"status": "success", "url": upload_to_gcs(out_path)}

        if task == "tts_emotion_text_custom":
            spk_path = save_temp(download_file(inp["spk_url"]))
            clear_mem()
            tts.infer(
                spk_audio_prompt=spk_path,
                text=inp["text"],
                emo_text=inp["emo_text"],
                emo_alpha=float(inp.get("emo_alpha", 0.6)),
                use_emo_text=True,
                use_random=inp.get("use_random", False),
                output_path=out_path
            )
            clear_mem()
            return {"status": "success", "url": upload_to_gcs(out_path)}

        if task == "merge":
            video = download_file(inp["video_url"])
            audio = download_file(inp["audio_url"])
            temp_dir = tempfile.mkdtemp()
            vid_path = os.path.join(temp_dir, "v.mp4")
            aud_path = os.path.join(temp_dir, "a.mp3")
            merged_path = os.path.join(temp_dir, "merged.mp4")
            with open(vid_path, "wb") as f: f.write(video)
            with open(aud_path, "wb") as f: f.write(audio)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", vid_path,
                "-i", aud_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                merged_path
            ], check=True)
            url = upload_to_gcs(merged_path, prefix="merged")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"status": "success", "url": url}

        return {"status": "error", "message": f"Unknown task: {task}"}

    except Exception as e:
        log(f"Error running task {task}: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- START SERVERLESS ----------------
runpod.serverless.start({"handler": handler})
