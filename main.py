#!/usr/bin/env python3
import os
import subprocess
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

@app.post("/api")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(400, "Upload a .txt file")

    tmp_path = None
    try:
        data = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Run the data analyst agent
        proc = subprocess.run(
            ["python", "data_analyst_agent.py", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=170
        )

        stdout = proc.stdout.decode("utf-8", errors="ignore")
        stderr = proc.stderr.decode("utf-8", errors="ignore")

        if proc.returncode != 0:
            message = f"Agent crashed (exit {proc.returncode})\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
            raise HTTPException(500, detail=message)

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as e:
            message = f"Invalid JSON from agent (decode error: {e})\nRaw STDOUT:\n{stdout}\nRaw STDERR:\n{stderr}"
            raise HTTPException(500, detail=message)

        return payload

    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Processing timeout (170s reached)")
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)