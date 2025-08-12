#!/usr/bin/env python3
import os
import subprocess
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "TDS Data Analyst Agent API"}

@app.post("/api")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        return {"error": "Please upload a .txt file"}

    tmp_path = None
    try:
        data = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Run the data analyst agent with 5-minute timeout
        proc = subprocess.run(
            ["python", "data_analyst_agent.py", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=290,  # 4 min 50 sec to allow for response processing
            env=os.environ.copy()
        )

        stdout = proc.stdout.decode("utf-8", errors="ignore")
        stderr = proc.stderr.decode("utf-8", errors="ignore")

        # Always try to return valid JSON, even if agent failed
        if proc.returncode != 0:
            # Try to parse any JSON output first
            try:
                payload = json.loads(stdout)
                return payload
            except json.JSONDecodeError:
                # Return fallback JSON structure
                return {"error": f"Agent failed with exit code {proc.returncode}"}

        try:
            payload = json.loads(stdout)
            return payload
        except json.JSONDecodeError as e:
            # Return fallback JSON structure instead of raising exception
            return {"error": f"Invalid JSON output: {str(e)}"}

    except subprocess.TimeoutExpired:
        # Return fallback JSON instead of raising exception
        return {"error": "Processing timeout (5 minutes exceeded)"}
    except Exception as e:
        # Catch any other unexpected errors and return valid JSON
        return {"error": f"Unexpected error: {str(e)}"}
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass  # Ignore cleanup errors