import os
import re
import base64
import subprocess
import tempfile
import asyncio
from typing import List, Dict, Tuple
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment.")

genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 8
OVERALL_TIMEOUT = 180  # Set to slightly less than 3 minutes (180s) to be safe
PER_SCRIPT_TIMEOUT = 120 # Timeout for a single code execution

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent (Gemini Edition)",
    description="An API that uses Google's Gemini model to autonomously perform data analysis.",
)

@app.get("/health", tags=["System"])
async def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})

# --- Core Agent Logic ---
async def execute_code(code_to_run: str, files: Dict[str, bytes]) -> Tuple[str, str]:
    """
    Asynchronously executes the given Python code in a sandboxed directory.
    This version is non-blocking and compatible with asyncio timeouts.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "wb") as f:
                f.write(content)

        script_path = os.path.join(temp_dir, "agent_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code_to_run)

        try:
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=PER_SCRIPT_TIMEOUT
            )
            return stdout_bytes.decode("utf-8", errors="ignore"), stderr_bytes.decode("utf-8", errors="ignore")
        except asyncio.TimeoutError:
            return "", f"The code execution timed out after {PER_SCRIPT_TIMEOUT} seconds."
        except Exception as e:
            return "", f"An unexpected error occurred during code execution: {e}"

def extract_python_code(llm_response: str) -> str:
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    if 'import ' in llm_response or 'print(' in llm_response:
        return llm_response.strip()
    return ""

def get_system_prompt():
    """
    Defines the LLM's role and instructions, encouraging a single, comprehensive script.
    """
    return """
You are an autonomous Python Data Analyst Agent. Your goal is to write a single, complete Python script to answer all parts of a user's request.

**Execution Plan:**
1.  **Analyze the ENTIRE request first.** Understand all calculations and visualizations required.
2.  **Plan your steps.** Think about the libraries you'll need (pandas, matplotlib, etc.).
3.  **Write ONE comprehensive script.** This script should perform all actions: load data, do all calculations, generate all plots, and print the final formatted JSON result. Do not solve the problem one piece at a time.
4.  Your final `print()` statement must be a single JSON object or array containing all the answers.

**CRITICAL RULE: To signal that you have finished the task, you MUST reply with the single word `OK` and nothing else. This is the only way to end the mission.**

---
**CRITICAL OUTPUT FORMATTING RULES:**
1.  Your final script must print a single, raw JSON object or array to standard output.
2.  Do not add any descriptive text or keys other than those explicitly requested.
3.  Study this example carefully:
    - USER'S QUESTION: "Return a JSON object with keys 'rows' and 'avg_price'. 1. How many rows? 2. What is the average price?"
    - **CORRECT SCRIPT:**
      ```python
      import pandas as pd
      import json
      df = pd.read_csv('data.csv')
      results = {
          'rows': len(df),
          'avg_price': df['price'].mean()
      }
      print(json.dumps(results))
      ```
4. Output format can be chanes as requested in the questions file / instructions.
---

**Your Instructions:**
1.  You will be given a question and a list of available data files (e.g., `sample-sales.csv`).
2.  You MUST write one Python script that solves all questions, following the rules above. The required libraries are pre-installed.
3.  If your code has an error, I will show you the error and you must provide the full, corrected script.
4.  If your code runs successfully, I will confirm it. You should then reply with `OK` if you believe your single script has fully solved the request.
"""

async def run_analysis_loop(question_text: str, data_files: Dict[str, bytes]):
    """
    This function contains the core agent logic with robust error handling for API responses.
    """
    file_list_str = ", ".join(data_files.keys()) if data_files else "None"
    
    safety_settings = [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    ]

    model = genai.GenerativeModel(
        LLM_MODEL, 
        system_instruction=get_system_prompt(),
        safety_settings=safety_settings
    )
    chat = model.start_chat(history=[])
    
    initial_prompt = f"Here is my question:\n---\n{question_text}\n---\nAvailable data files: [{file_list_str}]"
    feedback = initial_prompt
    last_successful_output = None

    for i in range(MAX_ITERATIONS):
        print(f"--- Iteration {i+1} ---")
        llm_response = ""

        try:
            response = await chat.send_message_async(feedback)
            if not response.parts:
                print("❌ LLM response was empty or blocked.")
                feedback = "Your previous response was blocked. Please analyze the original request again and provide the Python script."
                continue

            response_parts = [part.text for part in response.parts if hasattr(part, 'text')]
            llm_response = "".join(response_parts)

        except Exception as e:
            print(f"❌ An unexpected error occurred during the Gemini API call: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API call failed: {str(e)}")

        if not llm_response:
             print("⚠️ LLM response was parsed but resulted in an empty string. Asking agent to retry.")
             feedback = "Your previous response was empty. Please try generating the Python code again."
             continue

        if llm_response.strip().upper() == "OK":
            print("✅ LLM signaled completion.")
            if last_successful_output:
                return Response(content=last_successful_output, media_type="application/json")
            else:
                raise HTTPException(status_code=500, detail="Agent finished without producing any output.")

        code = extract_python_code(llm_response)
        if not code:
            feedback = "That was not code. Please provide a Python script or `OK` if finished."
            continue

        print(f"Executing code:\n{code[:350]}...")
        stdout, stderr = await execute_code(code, data_files)

        if stderr:
            print(f"Execution Error: {stderr}")
            last_error_line = stderr.strip().split('\n')[-1]
            feedback = (
                f"Your code produced an error: `{last_error_line}`. You MUST fix it. "
                "Analyze the error and provide the full, corrected Python script."
            )
        else:
            print(f"Execution Success. Output:\n{stdout[:350]}...")
            last_successful_output = stdout
            feedback = (
                "Your code ran successfully without any errors. "
                "Based on the script you wrote, is the task now complete? "
                "If YES, respond ONLY with `OK`. If NO, provide the next Python script."
            )
    
    raise HTTPException(status_code=500, detail=f"Agent could not complete the task in {MAX_ITERATIONS} iterations.")

@app.post("/api/", tags=["Data Analysis"])
async def analyze_data(request: Request):
    """
    The main API endpoint with a top-level timeout and corrected file handling.
    """
    try:
        form = await request.form()

        if "questions.txt" not in form:
            raise HTTPException(status_code=400, detail="questions.txt is a required field.")
            
        question_text = (await form['questions.txt'].read()).decode("utf-8")
        
        # ========== THE CRITICAL FIX IS HERE ==========
        # We iterate through the form items and use the KEY as the filename.
        # This preserves the meaningful name (e.g., 'sample-sales.csv') that the agent expects.
        data_files = {}
        for key, file_obj in form.items():
            if key != "questions.txt": # Exclude the question file from the sandbox
                data_files[key] = await file_obj.read()
        # ===============================================
        
        print(f"Starting analysis with a {OVERALL_TIMEOUT}-second timeout.")
        return await asyncio.wait_for(
            run_analysis_loop(question_text, data_files),
            timeout=OVERALL_TIMEOUT
        )

    except asyncio.TimeoutError:
        print(f"❌ Global timeout of {OVERALL_TIMEOUT} seconds reached. Terminating operation.")
        return JSONResponse(status_code=408, content=None)

    except Exception as e:
        print(f"An unexpected error occurred in the API handler: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)