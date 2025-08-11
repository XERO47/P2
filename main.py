import os
import re
import base64
import subprocess
import tempfile
import asyncio  # <-- Import asyncio
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import google.generativeai as genai
from google.generativeai import types

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file or environment.")

genai.configure(api_key=GOOGLE_API_KEY)

LLM_MODEL = "gemini-2.5-pro"
MAX_ITERATIONS = 7
OVERALL_TIMEOUT = 175  # Set to slightly less than 3 minutes (180s) to be safe
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
            # Use asyncio's non-blocking subprocess
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            # Wait for the subprocess to complete with its own timeout
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=PER_SCRIPT_TIMEOUT
            )
            return stdout_bytes.decode("utf-8", errors="ignore"), stderr_bytes.decode("utf-8", errors="ignore")
        except asyncio.TimeoutError:
            # This catches the timeout for a single script execution
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
    # System prompt remains the same
    return """
You are an autonomous Python Data Analyst Agent. Your sole purpose is to answer questions by writing and executing Python code.

**CRITICAL RULE: To signal that you have finished the task, you MUST reply with the single word `OK` and nothing else. This is the only way to end the mission.**

---
**CRITICAL OUTPUT FORMATTING RULES:**
1.  Your final output to be printed MUST be the raw data in the format requested (e.g., a JSON array).
2.  You MUST NOT add any descriptive text, labels, or keys. The output should be machine-readable values only.
3.  Study this example carefully:
    - USER'S QUESTION: "Answer with a JSON array: 1. How many rows? 2. What is the average price?"
    - **INCORRECT CODE:** `print('The number of rows is 50, and the average price is 99.50')`
    - **INCORRECT CODE:** `print(json.dumps({"rows": 50, "average_price": 99.50}))`
    - **CORRECT CODE:** `print(json.dumps([50, 99.50]))`
---

**Your Instructions:**
1.  You will be given a question and a list of available data files.  
    Before writing code, infer the column names from the data file provided.
2.  You MUST write Python code to answer the question, strictly following the output format rules above.
3.  You can use common data science libraries like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`, `beautifulsoup4`. They are pre-installed.
4.  If you need to generate a plot, save it to a file (e.g., `plot.png`) and then print its base64-encoded data URI to stdout.
5.  You operate in a loop. I will execute your code and give you the result.
    - If you get an **error (stderr)**, you MUST analyze it and provide the corrected, full Python code.
    - If your code runs **successfully**, I will inform you. You must then decide if the task is complete.
        - If YES, respond with `OK`.
        - If NO (e.g., more steps are needed), provide the **next** Python code block.
6.  Do not add any explanations or comments. Your response must be either a Python code block or the word `OK`.
7.  DON'T ASSUME ANYTHING GET EVERTHING AS YOU NEED.
"""

async def run_analysis_loop(question_text: str, data_files: Dict[str, bytes]):
    """
    This function contains the core agent logic with robust error handling for all API responses.
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
        llm_response = "" # Initialize empty response string

        try:
            response = await chat.send_message_async(feedback)

            # --- NEW: ROBUST RESPONSE PARSING TO HANDLE MULTI-PART RESPONSES ---
            # This logic replaces the fragile `response.text` accessor.

            if not response.parts:
                print("❌ LLM response was empty or blocked (likely a safety filter).")
                feedback = (
                    "Your previous response was blocked. Please analyze the original request again and provide the Python script."
                )
                continue

            # Iterate through all parts and safely concatenate their text content.
            # This correctly handles both single-part and multi-part responses.
            response_parts = []
            for part in response.parts:
                try:
                    # Each part should have a 'text' attribute.
                    response_parts.append(part.text)
                except ValueError:
                    # This part might not be text, so we'll log it and skip.
                    print(f"⚠️ Skipping a non-text part in the LLM response: {part}")
                    continue
            
            llm_response = "".join(response_parts)
            # --- END OF NEW PARSING LOGIC ---

        except Exception as e:
            print(f"❌ An unexpected error occurred during the Gemini API call: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API call failed: {str(e)}")

        if not llm_response:
             print("⚠️ LLM response was parsed but resulted in an empty string. Asking agent to retry.")
             feedback = "Your previous response was empty after parsing. Please try generating the Python code again."
             continue

        if llm_response.strip().upper() == "OK":
            print("✅ LLM signaled completion. Responding with last successful output.")
            if last_successful_output:
                return Response(content=last_successful_output, media_type="application/json")
            else:
                raise HTTPException(status_code=500, detail="Agent finished without producing any output.")

        code = extract_python_code(llm_response)
        if not code:
            feedback = "That was not code. Please provide only a Python code block to solve the task or `OK` if you are finished."
            continue

        print(f"Executing code:\n{code[:350]}...")
        stdout, stderr = await execute_code(code, data_files)

        if stderr:
            print(f"Execution Error: {stderr}")
            last_error_line = stderr.strip().split('\n')[-1]
            feedback = (
                f"Your code produced an error. You MUST fix it.\n"
                f"The specific error was: `{last_error_line}`\n"
                f"Please analyze this error and provide the full, corrected Python script."
            )
        else:
            print(f"Execution Success. Output:\n{stdout[:350]}...")
            last_successful_output = stdout
            feedback = (
                "Your code ran successfully without any errors. "
                "Based on the code you just wrote, do you believe this is the final, complete answer? "
                "If YES, respond ONLY with `OK`. If NO, provide the next Python code."
            )
    
    raise HTTPException(status_code=500, detail=f"Agent could not complete the task in {MAX_ITERATIONS} iterations.")

@app.post("/api/", tags=["Data Analysis"])
async def analyze_data(request: Request):
    """
    The main API endpoint with a top-level timeout.
    """
    try:
        form = await request.form()
        uploaded_files: Dict[str, UploadFile] = {k: v for k, v in form.items()}

        if "questions.txt" not in uploaded_files:
            raise HTTPException(status_code=400, detail="questions.txt is a required field.")
            
        question_text = (await uploaded_files.pop("questions.txt").read()).decode("utf-8")
        
        data_files = {}
        for filename, file_obj in uploaded_files.items():
            data_files[file_obj.filename] = await file_obj.read()
        
        # --- Run the agent logic with a hard timeout ---
        print(f"Starting analysis with a {OVERALL_TIMEOUT}-second timeout.")
        return await asyncio.wait_for(
            run_analysis_loop(question_text, data_files),
            timeout=OVERALL_TIMEOUT
        )

    except asyncio.TimeoutError:
        print(f"❌ Global timeout of {OVERALL_TIMEOUT} seconds reached. Terminating operation.")
        # Return a null JSON response with a 408 Request Timeout status code
        return JSONResponse(status_code=408, content=None)

    except Exception as e:
        print(f"An unexpected error occurred in the API handler: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)