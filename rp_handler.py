import os
import gc
import uuid
import tempfile
import subprocess
import shutil
import torch
import runpod
import requests
from datetime import timedelta
from indextts.infer_v2 import IndexTTS2
import boto3
from huggingface_hub import snapshot_download

# -------------------- CONFIG --------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "runpodstorageforserverless")

log_prefix = "[DEBUG]"

# -------------------- Setup --------------------
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

# -------------------- AWS S3 CLIENT --------------------
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_to_s3(local_path, prefix="tts_outputs"):
    """
    Upload a local file to S3 under the specified prefix and return the public URL.
    Bucket must allow public read via policy.
    """
    try:
        key = f"{prefix}/{uuid.uuid4()}{os.path.splitext(local_path)[1]}"
        s3.upload_file(
            Filename=local_path,
            Bucket=BUCKET_NAME,
            Key=key,
            ExtraArgs={"ContentType": "audio/mpeg"}  # Adjust MIME if needed
        )
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"
        log(f"Uploaded to S3: {url}")
        return url
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

# ---------------- INIT MODEL ----------------
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False
)

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
            return {"status": "success", "url": upload_to_s3(out_path)}

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
            return {"status": "success", "url": upload_to_s3(out_path)}

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
            return {"status": "success", "url": upload_to_s3(out_path)}

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
            return {"status": "success", "url": upload_to_s3(out_path)}

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
            return {"status": "success", "url": upload_to_s3(out_path)}

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
            url = upload_to_s3(merged_path, prefix="merged")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"status": "success", "url": url}

        return {"status": "error", "message": f"Unknown task: {task}"}

    except Exception as e:
        log(f"Error running task {task}: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- START SERVERLESS ----------------
runpod.serverless.start({"handler": handler})
