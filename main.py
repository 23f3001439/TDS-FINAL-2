

#!/usr/bin/env python3
import os
import sys
import json
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException, Response

app = FastAPI()

# Health/Root so Render/browser don’t 404
@app.get("/")
def root():
    return {"status": "ok"}

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/api")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(400, "Upload a .txt file")

    # Optional hard guard if your agent needs this
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY is not set")

    tmp_path = None
    try:
        data = await file.read()

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Use the same Python interpreter Render used to install deps
        proc = subprocess.run(
            [sys.executable, "data_analyst_agent.py", tmp_path],
            capture_output=True,
            text=True,
            timeout=170,
        )

        if proc.returncode != 0:
            raise HTTPException(
                500,
                detail=f"Agent crashed (exit {proc.returncode})\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise HTTPException(
                500,
                detail=f"Invalid JSON from agent (decode error: {e})\nRaw STDOUT:\n{proc.stdout}\nRaw STDERR:\n{proc.stderr}"
            )

        return payload

    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Processing timeout (170s reached)")

    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)


