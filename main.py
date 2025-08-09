import os
import re
import base64
import subprocess
import tempfile
from typing import List, Dict, Tuple

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response
import google.generativeai as genai

# --- Configuration ---
# It's best practice to set the API key as an environment variable
# export GOOGLE_API_KEY='your_api_key_here'
GOOGLE_API_KEY="AIzaSyAx8XgMBtfkbd4j28Pnauv2TDQ_QPOWYE4"

genai.configure(api_key=GOOGLE_API_KEY)

# Use a powerful model that's good at following instructions and coding.
# gemini-1.5-pro-latest is a great choice.
LLM_MODEL = "gemini-2.5-pro" 
MAX_ITERATIONS = 7 # Safety break to prevent infinite loops and stay under 3 mins

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent (Gemini Edition)",
    description="An API that uses Google's Gemini model to autonomously perform data analysis.",
)


# --- Core Agent Logic ---

def execute_code(code_to_run: str, files: Dict[str, bytes]) -> Tuple[str, str]:
    """
    Executes the given Python code in a sandboxed temporary directory.
    - Creates a temporary directory.
    - Writes the provided data files to this directory.
    - Writes the Python code to a script file.
    - Executes the script using subprocess.
    - Returns the stdout and stderr of the execution.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files.items():
            with open(os.path.join(temp_dir, filename), "wb") as f:
                f.write(content)

        script_path = os.path.join(temp_dir, "agent_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code_to_run)

        try:
            # Execute with a timeout to respect the 3-minute limit
            process = subprocess.run(
                ["python", script_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120, # 2-minute timeout for the script itself
                encoding="utf-8",
                errors="ignore"
            )
            return process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            return "", "The code execution timed out after 120 seconds."


def extract_python_code(llm_response: str) -> str:
    """
    Extracts Python code from a markdown-formatted string.
    """
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no markdown block, check if the response itself is valid code.
    # A simple heuristic: if it contains 'import' or 'print'.
    if 'import ' in llm_response or 'print(' in llm_response:
        return llm_response.strip()
    return ""


def get_system_prompt():
    """Defines the LLM's role and instructions for Gemini."""
    return """
You are an autonomous Python Data Analyst Agent. Your sole purpose is to answer questions by writing and executing Python code.

**Your Instructions:**
1.  You will be given a question and a list of available data files.
2.  You MUST write Python code to answer the question. Your code must be complete and runnable.
3.  **Your code MUST print the final answer to standard output.** The format of the final answer (e.g., JSON array, JSON object) will be specified in the user's question. Adhere to this format strictly.
4.  You can use common data science libraries like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`, `beautifulsoup4`. They are pre-installed.
5.  If you need to generate a plot, save it to a file (e.g., `plot.png`) and then print its base64-encoded data URI to stdout. For example: `print(f"data:image/png;base64,{base64_string}")`.
6.  You operate in a loop. I will execute your code and give you the result (stdout or stderr).
    - If you get an **error (stderr)**, you MUST analyze it and provide the corrected, full Python code. Do not apologize or explain, just provide the fixed code.
    - If you get **output (stdout)**, you MUST check if it is the complete and final answer in the correct format.
        - If YES, your **ONLY** response must be the word `OK`. Nothing else.
        - If NO (e.g., it's intermediate output), you MUST provide the **next** block of Python code to continue the analysis.
7.  Do not add any explanations, comments, or apologies. Your response must be either a Python code block or the word `OK`.
"""


@app.post("/api/")
async def analyze_data(request: Request):
    """
    The main API endpoint to handle data analysis requests.
    """
    try:
        form = await request.form()
        uploaded_files: Dict[str, UploadFile] = {k: v for k, v in form.items()}

        if "questions.txt" not in uploaded_files:
            raise HTTPException(status_code=400, detail="questions.txt is a required field.")
            
        question_text = (await uploaded_files.pop("questions.txt").read()).decode("utf-8")
        
        data_files = {}
        for filename, file_obj in uploaded_files.items():
            # The key from the form is often the field name, not the filename.
            # We use `file_obj.filename` to get the actual file name.
            data_files[file_obj.filename] = await file_obj.read()
        
        file_list_str = ", ".join(data_files.keys()) if data_files else "None"

        # --- The Agent's Self-Correction Loop ---
        model = genai.GenerativeModel(LLM_MODEL, system_instruction=get_system_prompt())
        chat = model.start_chat(history=[])
        
        # Initial prompt to kick off the process
        initial_prompt = f"Here is my question:\n---\n{question_text}\n---\nAvailable data files: [{file_list_str}]"
        feedback = initial_prompt
        last_successful_output = None

        for i in range(MAX_ITERATIONS):
            print(f"--- Iteration {i+1} ---")

            # 1. Ask LLM to generate code based on the current state (feedback)
            response = chat.send_message(feedback)
            llm_response = response.text
            
            # 2. Check for the "OK" signal to terminate
            if llm_response.strip().upper() == "OK":
                print("LLM signaled completion. Responding with last successful output.")
                if last_successful_output:
                    # The output is expected to be a string, potentially JSON formatted.
                    # Returning it as raw content with application/json lets the client parse it.
                    return Response(content=last_successful_output, media_type="application/json")
                else:
                    raise HTTPException(status_code=500, detail="Agent finished without producing any output.")

            # 3. Extract and execute the code
            code = extract_python_code(llm_response)
            if not code:
                # If LLM gives an explanation instead of code, try to steer it back.
                feedback = "That was not code. Please provide only a Python code block to solve the task or `OK` if you are finished."
                continue

            print(f"Executing code:\n{code[:350]}...")
            stdout, stderr = execute_code(code, data_files)

            # 4. Prepare feedback for the next loop iteration
            if stderr:
                print(f"Execution Error: {stderr}")
                feedback = f"Your code produced an error. You MUST fix it.\nError:\n---\n{stderr}\n---\nProvide the full, corrected Python code."
            else:
                print(f"Execution Success. Output:\n{stdout[:350]}...")
                last_successful_output = stdout
                feedback = f"Your code ran successfully. Here is the output:\n---\n{stdout}\n---\nIs this the final answer in the correct format? If yes, respond ONLY with `OK`. If no, provide the next block of Python code to continue the analysis."
        
        raise HTTPException(status_code=500, detail=f"Agent could not complete the task in {MAX_ITERATIONS} iterations.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Provide a more detailed error response for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # To run this server:
    # 1. Install dependencies: pip install "fastapi[all]" google-generativeai pandas numpy scikit-learn matplotlib requests beautifulsoup4
    # 2. Get a Google API Key from Google AI Studio.
    # 3. Set your Google API Key: export GOOGLE_API_KEY='your-key'
    # 4. Run the server: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)