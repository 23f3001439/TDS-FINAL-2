#!/usr/bin/env python3
import os, sys, json, asyncio, logging, re, time
import pandas as pd, requests, duckdb, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
from scipy import stats
import google.generativeai as genai
from fallback_templates import get_fallback_template

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
MODEL = "models/gemini-2.5-flash"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def detect_task_patterns(text: str) -> dict:
    """Analyze the task to understand data sources, output format, and analysis type"""
    patterns = {
        'data_sources': [],
        'output_format': 'json_array',
        'has_visualization': False,
        'analysis_type': 'general',
        'specific_requirements': []
    }
    
    text_lower = text.lower()
    
    # Detect data sources
    if 'wikipedia' in text_lower or 'wiki' in text_lower:
        patterns['data_sources'].append('wikipedia')
    if 'duckdb' in text_lower or 'parquet' in text_lower or 's3://' in text:
        patterns['data_sources'].append('duckdb')
    if '.csv' in text_lower:
        patterns['data_sources'].append('csv')
    
    # Detect output format
    if 'json object' in text_lower or 'json dictionary' in text_lower:
        patterns['output_format'] = 'json_object'
    elif 'json array' in text_lower:
        patterns['output_format'] = 'json_array'
    
    # Detect visualization requirements
    if any(word in text_lower for word in ['plot', 'chart', 'graph', 'scatter', 'histogram', 'visualization']):
        patterns['has_visualization'] = True
    
    # Detect analysis type
    if 'correlation' in text_lower:
        patterns['analysis_type'] = 'correlation'
    elif 'regression' in text_lower:
        patterns['analysis_type'] = 'regression'
    elif 'count' in text_lower or 'how many' in text_lower:
        patterns['analysis_type'] = 'counting'
    elif 'earliest' in text_lower or 'latest' in text_lower or 'first' in text_lower:
        patterns['analysis_type'] = 'temporal'
    
    # Extract specific requirements
    if 'base64' in text_lower or 'data uri' in text_lower:
        patterns['specific_requirements'].append('base64_encoding')
    if '100' in text and ('kb' in text_lower or 'kilobyte' in text_lower):
        patterns['specific_requirements'].append('size_limit_100kb')
    if 'dotted' in text_lower and 'red' in text_lower:
        patterns['specific_requirements'].append('dotted_red_line')
    
    return patterns

async def plan_task(text: str) -> str:
    """Generate Python code using Gemini"""
    patterns = detect_task_patterns(text)
    
    prompt = f"""You are an expert data analyst. Generate ONLY Python code (no markdown, no explanations) that:

TASK ANALYSIS:
- Data sources: {patterns['data_sources']}
- Output format: {patterns['output_format']}
- Visualization needed: {patterns['has_visualization']}
- Analysis type: {patterns['analysis_type']}

REQUIRED IMPORTS:
import pandas as pd
import requests
import duckdb
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import json
from scipy import stats
import seaborn as sns
plt.switch_backend('Agg')

CRITICAL RULES:
1. Always handle errors gracefully with try/except
2. Cast columns to appropriate types before operations
3. For string operations: df['col'].astype(str) first
4. Keep images under 100KB (use dpi=80, figsize=(8,6))
5. Assign final answer to variable named `result`
6. For Wikipedia: Use pd.read_html(url) and select appropriate table
7. For DuckDB: Use duckdb.sql(query).df()
8. For visualizations: Save as PNG with base64 encoding
9. For dotted red lines: Use 'r--' style

TASK:
\"\"\"{text}\"\"\"

Generate ONLY the Python code that solves this task.
"""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(
                MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4000
                )
            )
            
            resp = model.generate_content(prompt.strip())
            code = resp.text.strip()
            
            # Clean up markdown formatting
            if code.startswith("```"):
                code = re.sub(r"^```(?:python)?\n", "", code)
                code = re.sub(r"\n```$", "", code).strip()
            
            return code
            
        except Exception as e:
            logging.warning(f"Gemini attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(1.5 ** attempt)

def sanitize(obj):
    """Sanitize data for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    elif isinstance(obj, list):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return obj

def execute_code(code: str):
    """Execute generated code"""
    ns = {
        "pd": pd,
        "requests": requests,
        "duckdb": duckdb,
        "plt": plt,
        "io": io,
        "base64": base64,
        "np": np,
        "json": json,
        "stats": stats,
        "sns": sns,
        "time": time,
        "re": re,
        "os": os
    }
    
    try:
        exec(code, ns)
        
        if "result" not in ns:
            raise RuntimeError("Generated code did not assign `result` variable")

        result = sanitize(ns["result"])
        
        # Validate JSON serializability
        json.dumps(result)
        
        # Check image size constraints
        def check_image_size(obj):
            if isinstance(obj, str) and obj.startswith("data:image/"):
                size_bytes = len(obj.encode('utf-8'))
                if size_bytes > 100_000:
                    raise RuntimeError(f"Image exceeds 100KB limit: {size_bytes/1024:.1f}KB")
            elif isinstance(obj, list):
                for item in obj:
                    check_image_size(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    check_image_size(value)
        
        check_image_size(result)
        return result
        
    except Exception as e:
        logging.error(f"Code execution failed: {str(e)}")
        return {"error": f"Execution failed: {str(e)}"}

async def execute_with_retry(task: str, max_attempts: int = 2) -> dict:
    """Execute task with retry and self-correction"""
    
    for attempt in range(max_attempts):
        try:
            logging.info(f"Planning attempt {attempt + 1} with Gemini...")
            code = await plan_task(task)
            
            if attempt == 0:
                print("=== GEMINI GENERATED CODE ===", file=sys.stderr)
                print(code, file=sys.stderr)
                print("=============================", file=sys.stderr)

            logging.info("Executing generated code...")
            result = await asyncio.get_event_loop().run_in_executor(None, execute_code, code)
            
            if not (isinstance(result, dict) and "error" in result):
                return {"success": True, "result": result, "attempt": attempt + 1}
            
            if attempt < max_attempts - 1:
                logging.warning(f"Attempt {attempt + 1} failed: {result['error']}")
                logging.info("Attempting self-correction...")
                
                task = f"""
PREVIOUS ATTEMPT FAILED WITH ERROR: {result['error']}

Please fix the issue and try a different approach. Common fixes:
- Check data availability and column names
- Handle missing data gracefully
- Use proper data types and casting
- Add error handling for network requests
- Verify table selection for HTML parsing

ORIGINAL TASK:
{task}
"""
            else:
                return {"success": False, "error": result['error'], "attempt": attempt + 1}
                
        except Exception as e:
            error_msg = f"Planning failed on attempt {attempt + 1}: {str(e)}"
            logging.error(error_msg)
            
            if attempt == max_attempts - 1:
                return {"success": False, "error": error_msg, "attempt": attempt + 1}
    
    return {"success": False, "error": "Max attempts exceeded", "attempt": max_attempts}

async def main():
    if len(sys.argv) != 2:
        print("Usage: data_analyst_agent.py <question.txt>")
        sys.exit(1)

    start_time = time.time()
    task = open(sys.argv[1], encoding="utf-8").read().strip()
    
    patterns = detect_task_patterns(task)
    logging.info(f"Task analysis: {patterns}")
    
    try:
        execution_result = await execute_with_retry(task)
        
        if execution_result["success"]:
            result = execution_result["result"]
            total_time = time.time() - start_time
            logging.info(f"✅ Task completed successfully in {total_time:.2f}s")
            
            json.dump(result, sys.stdout)
            sys.stdout.write("\n")
            
        else:
            logging.warning(f"Primary execution failed: {execution_result['error']}")
            logging.info("Attempting fallback template...")
            
            try:
                fallback_code = get_fallback_template(task)
                
                print("=== FALLBACK TEMPLATE CODE ===", file=sys.stderr)
                print(fallback_code, file=sys.stderr)
                print("===============================", file=sys.stderr)
                
                result = await asyncio.get_event_loop().run_in_executor(None, execute_code, fallback_code)
                
                if isinstance(result, dict) and "error" in result:
                    raise Exception(result["error"])
                
                total_time = time.time() - start_time
                logging.info(f"✅ Fallback completed successfully in {total_time:.2f}s")
                json.dump(result, sys.stdout)
                sys.stdout.write("\n")
                
            except Exception as fallback_error:
                total_time = time.time() - start_time
                final_error = f"Both primary and fallback failed after {total_time:.2f}s. Primary: {execution_result['error']} | Fallback: {str(fallback_error)}"
                logging.error(final_error)
                json.dump({"error": final_error}, sys.stdout)
                sys.stdout.write("\n")
                sys.exit(1)
                
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Critical failure after {total_time:.2f}s: {str(e)}"
        logging.error(error_msg)
        json.dump({"error": error_msg}, sys.stdout)
        sys.stdout.write("\n")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())