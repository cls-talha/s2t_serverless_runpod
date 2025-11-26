FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt update && apt install -y ffmpeg git-lfs
RUN git lfs install

RUN git clone https://github.com/index-tts/index-tts.git /workspace/index-tts

WORKDIR /workspace/index-tts

RUN git lfs pull

COPY rp_handler.py /workspace/index-tts/rp_handler.py

# Install uv and dependencies inside virtual env
RUN pip install -U uv
RUN uv sync --all-extras
RUN uv pip install "huggingface-hub[cli,hf_xet]" runpod google-cloud-storage protobuf==3.20.* hf_transfer

ENV PYTHONPATH="$PYTHONPATH:."
ENV VIRTUAL_ENV=/workspace/index-tts/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["/bin/bash", "-c", "source .venv/bin/activate && python -u rp_handler.py"]
