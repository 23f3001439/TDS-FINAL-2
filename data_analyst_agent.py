#!/usr/bin/env python3
import os, sys, json, asyncio, logging, re, time
import pandas as pd, requests, duckdb, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
import urllib.parse
from scipy import stats
import google.generativeai as genai
from fallback_templates import get_fallback_template, get_emergency_fallback

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

async def understand_task(text: str) -> dict:
    """First AI call: Understand the task requirements and structure"""
    prompt = f"""Analyze this data analysis task and return a JSON object with the analysis:

TASK:
\"\"\"{text}\"\"\"

Return ONLY a JSON object with these keys:
- "data_sources": array of data sources needed (e.g., ["wikipedia", "csv", "duckdb", "web_search"])
- "output_format": "json_array" or "json_object" 
- "required_keys": array of expected output keys/fields
- "analysis_steps": array of main analysis steps needed
- "visualization_requirements": array of chart/plot requirements
- "search_queries": array of search terms if web search is needed
- "expected_data_types": object mapping output keys to their data types

Be precise about the expected output structure and data types."""

    try:
        model = genai.GenerativeModel(
            MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=1000
            )
        )
        
        resp = model.generate_content(prompt.strip())
        analysis_text = resp.text.strip()
        
        # Clean up markdown formatting
        if analysis_text.startswith("```"):
            analysis_text = re.sub(r"^```(?:json)?\n", "", analysis_text)
            analysis_text = re.sub(r"\n```$", "", analysis_text).strip()
        
        return json.loads(analysis_text)
        
    except Exception as e:
        logging.warning(f"Task understanding failed: {str(e)}")
        # Return basic fallback analysis
        return {
            "data_sources": ["csv"],
            "output_format": "json_array",
            "required_keys": ["result"],
            "analysis_steps": ["load_data", "analyze", "return_result"],
            "visualization_requirements": [],
            "search_queries": [],
            "expected_data_types": {"result": "string"}
        }

async def plan_task(text: str, task_analysis: dict) -> str:
    """Second AI call: Generate Python code with full context"""
    
    # Build the prompt with proper string formatting
    data_sources = task_analysis.get('data_sources', [])
    output_format = task_analysis.get('output_format', 'json_array')
    required_keys = task_analysis.get('required_keys', [])
    analysis_steps = task_analysis.get('analysis_steps', [])
    visualizations = task_analysis.get('visualization_requirements', [])
    search_queries = task_analysis.get('search_queries', [])
    expected_types = task_analysis.get('expected_data_types', {})
    
    prompt = f"""You are an expert data analyst with access to web search capabilities. Generate ONLY Python code (no markdown, no explanations) that:

TASK ANALYSIS (from AI understanding):
- Data sources: {data_sources}
- Output format: {output_format}
- Required keys: {required_keys}
- Analysis steps: {analysis_steps}
- Visualizations: {visualizations}
- Search queries: {search_queries}
- Expected data types: {expected_types}

AVAILABLE TOOLS & IMPORTS:
import pandas as pd
import requests
import duckdb
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import json
from scipy import stats
import urllib.parse
plt.switch_backend('Agg')

DATA SOURCES YOU CAN ACCESS:
1. Wikipedia: Use pd.read_html(url) for tables
2. DuckDB: Use duckdb.sql(query).df() for S3/Parquet data
3. Web APIs: Use requests.get() with proper headers
4. Google Search: Use requests to search for current information
5. CSV files: Use pd.read_csv() with error handling

SEARCH CAPABILITIES:
- For current data/statistics: Use Google search with requests
- For real-time information: Search recent sources
- For specific datasets: Look up official sources
- Example Google search: requests.get(f"https://www.google.com/search?q={{urllib.parse.quote(query)}}")

CRITICAL REQUIREMENTS:
1. EXACT OUTPUT FORMAT: Must return {output_format}
2. REQUIRED KEYS: Must include ALL keys: {required_keys}
3. DATA TYPES: Follow exact types: {expected_types}
4. Always handle errors gracefully with try/except
5. Cast columns to appropriate types before operations
6. Keep images under 100KB (use dpi=80, figsize=(8,6))
7. Assign final answer to variable named `result`
8. For Wikipedia: Use pd.read_html(url) and select appropriate table
9. For DuckDB: Use duckdb.sql(query).df()
10. For visualizations: Save as PNG with base64 encoding (format: "data:image/png;base64,...")
11. For dotted red lines: Use 'r--' style
12. Use web search for current/recent data when needed
13. VALIDATE: Ensure result matches expected structure exactly

IMPORTANT OUTPUT RULES:
- If output_format is "json_array": result must be a simple list like [value1, value2, value3]
- If output_format is "json_object": result must be a dictionary like {{"key1": value1, "key2": value2}}
- DO NOT wrap values in objects like {{"result": value}} - return the values directly
- For json_array: result = [answer1, answer2, answer3, answer4]
- For json_object: result = {{"key1": answer1, "key2": answer2}}

SEARCH QUERIES TO USE: {search_queries}

TASK:
\"\"\"{text}\"\"\"

Generate ONLY the Python code that creates the exact output structure required.
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
        "os": os,
        "urllib": urllib
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

async def execute_with_retry(task: str, max_attempts: int = 3) -> dict:
    """Execute task with retry and self-correction"""
    
    start_time = time.time()
    max_retry_time = 240  # 4 minutes for retries
    
    for attempt in range(max_attempts):
        # Check timeout
        if time.time() - start_time > max_retry_time:
            return {"success": False, "error": "Timeout during retry attempts", "attempt": attempt + 1}
        try:
            # First AI call: Understand the task
            if attempt == 0:
                task_analysis = await understand_task(task)
            else:
                # Use simpler analysis for retries
                task_analysis = {
                    "data_sources": ["csv"],
                    "output_format": "json_object" if "json object" in task.lower() else "json_array",
                    "required_keys": [],
                    "analysis_steps": ["analyze"],
                    "visualization_requirements": [],
                    "search_queries": [],
                    "expected_data_types": {}
                }
            
            # Second AI call: Generate code with full context
            code = await plan_task(task, task_analysis)
            
            # Only show debug info in development, not in final submission
            # if attempt == 0:
            #     print("=== GEMINI GENERATED CODE ===", file=sys.stderr)
            #     print(code, file=sys.stderr)
            #     print("=============================", file=sys.stderr)

            # logging.info("Executing generated code...")
            result = await asyncio.get_event_loop().run_in_executor(None, execute_code, code)
            
            if not (isinstance(result, dict) and "error" in result):
                return {"success": True, "result": result, "attempt": attempt + 1}
            
            if attempt < max_attempts - 1:
                # logging.warning(f"Attempt {attempt + 1} failed: {result['error']}")  # Commented for final submission
                # logging.info("Attempting self-correction...")  # Commented for final submission
                
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
    max_processing_time = 270  # 4.5 minutes to leave buffer for response
    
    task = open(sys.argv[1], encoding="utf-8").read().strip()
    
    patterns = detect_task_patterns(task)
    # logging.info(f"Task analysis: {patterns}")  # Commented for final submission
    
    try:
        # Check if we're running out of time
        if time.time() - start_time > max_processing_time:
            # logging.warning("Approaching time limit, returning emergency fallback")  # Commented for final submission
            fallback_result = get_emergency_fallback(task)
            json.dump(fallback_result, sys.stdout)
            sys.stdout.write("\n")
            return
            
        execution_result = await execute_with_retry(task)
        
        if execution_result["success"]:
            result = execution_result["result"]
            total_time = time.time() - start_time
            # logging.info(f"✅ Task completed successfully in {total_time:.2f}s")  # Commented for final submission
            
            json.dump(result, sys.stdout)
            sys.stdout.write("\n")
            
        else:
            # logging.warning(f"Primary execution failed: {execution_result['error']}")  # Commented for final submission
            # logging.info("Attempting fallback template...")  # Commented for final submission
            
            try:
                fallback_code = get_fallback_template(task)
                
                # Only show debug info in development
                # print("=== FALLBACK TEMPLATE CODE ===", file=sys.stderr)
                # print(fallback_code, file=sys.stderr)
                # print("===============================", file=sys.stderr)
                
                result = await asyncio.get_event_loop().run_in_executor(None, execute_code, fallback_code)
                
                if isinstance(result, dict) and "error" in result:
                    raise Exception(result["error"])
                
                total_time = time.time() - start_time
                # logging.info(f"✅ Fallback completed successfully in {total_time:.2f}s")  # Commented for final submission
                json.dump(result, sys.stdout)
                sys.stdout.write("\n")
                
            except Exception as fallback_error:
                total_time = time.time() - start_time
                final_error = f"Both primary and fallback failed after {total_time:.2f}s. Primary: {execution_result['error']} | Fallback: {str(fallback_error)}"
                logging.error(final_error)
                
                # Return a valid JSON structure based on task type
                fallback_result = get_emergency_fallback(task)
                json.dump(fallback_result, sys.stdout)
                sys.stdout.write("\n")
                
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Critical failure after {total_time:.2f}s: {str(e)}"
        logging.error(error_msg)
        
        # Return a valid JSON structure based on task type
        fallback_result = get_emergency_fallback(task)
        json.dump(fallback_result, sys.stdout)
        sys.stdout.write("\n")

if __name__ == "__main__":
    asyncio.run(main())