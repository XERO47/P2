import os
import re
import base64
import subprocess
import tempfile
import asyncio
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from google import genai
from google.genai import types

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found. Please set it in your .env file or environment.")

# Initialize the Google GenAI client
client = genai.Client(api_key=GOOGLE_API_KEY)

LLM_MODEL = "gemini-2.5-flash"
MAX_ITERATIONS = 8
OVERALL_TIMEOUT = 180  # Set to slightly less than 3 minutes (180s) to be safe
PER_SCRIPT_TIMEOUT = 120 # Timeout for a single code execution

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent (Google GenAI SDK) with Image Support",
    description="An API that uses Google's GenAI SDK to autonomously perform data analysis including image analysis.",
)

@app.get("/health", tags=["System"])
async def health_check():
    return JSONResponse(status_code=200, content={"status": "ok"})

# --- Helper Functions ---
def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_FORMATS

def encode_image_to_base64(image_data: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_data).decode('utf-8')

def get_image_mime_type(filename: str) -> str:
    """Get MIME type for image based on file extension."""
    ext = Path(filename).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')

# --- Core Agent Logic ---
async def execute_code(code_to_run: str, files: Dict[str, bytes]) -> Tuple[str, str]:
    """
    Asynchronously executes the given Python code in a sandboxed directory.
    This version is non-blocking and compatible with asyncio timeouts.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files.items():
            # Only save non-image files to the temp directory for code execution
            if not is_image_file(filename):
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
    Defines the LLM's role and instructions including image analysis capabilities.
    """
    return """


You are an autonomous Python Data Analyst Agent with image analysis capabilities. You will solve problems by following a mandatory two-step process: **1. Inspection**, then **2. Analysis**.

**CRITICAL ANTI-HALLUCINATION RULES:**

- You MUST execute Python code to access real data - NEVER guess or fabricate numbers.
- If code fails, you MUST fix the code and re-execute - do NOT provide estimated results.
- Only provide final JSON output after successful code execution with real data.
- NEVER use phrases like "based on typical data" or "approximately" or "estimated".
- All numbers, calculations, and insights must come from actual code execution.

---

**STEP 1: DATA INSPECTION (Your First Action)**
Your first task is to understand the data you've been given. You MUST NOT guess column names or image contents.

* **For data files (CSV, Excel, etc.):**

  1. Write a simple script to read the header of EACH provided data file and print its columns.
  2. This is the only code you should write for data files in your first turn.
* **For image files:**

  1. I will provide you with the image(s) directly in the conversation.
  2. Analyze what you can see in the image(s) and describe their contents. This is a text-based analysis; do not write code for this part.

  **Example Inspection Script (if given `sample-sales.csv`):**

  ```python
  import pandas as pd
  try:
      df = pd.read_csv('sample-sales.csv')
      print("Columns for 'sample-sales.csv':", df.columns.tolist())
  except Exception as e:
      print(f"Error inspecting file 'sample-sales.csv': {e}")
  ```

---

**STEP 2: DATA ANALYSIS (Your Second Action)**
After you receive the column names from data files and have analyzed any images, you will have the necessary information to write the full analysis script.

* **For data analysis tasks:**

  1. **Analyze the ENTIRE request.** Understand all calculations and visualizations required.
  2. **Write ONE comprehensive script.** This script must perform all actions: load the data using the correct column names, perform all calculations, generate all plots, and print the final formatted JSON result.
  3. Your final `print()` statement must be a single JSON object or array containing all the answers.
* **For image analysis tasks:**

  1. If the task requires you to extract information from an image, integrate your observations into the final output.
  2. If combining image analysis with data analysis, use insights from both sources.
If the question ask for web scrapping you can request data from python code analyse it your self and output using a python code.
---

**CRITICAL RULE: To signal that you have finished the task after the Analysis step, you MUST reply with the single word `OK` and nothing else. This is the only way to end the mission.**

---

**Available Tools and Libraries:**
You MUST use only the following libraries to solve the tasks. Do not attempt to import or install any other packages.

* **Core Data Analysis:** `pandas`, `numpy`, `scipy`, `duckdb` (for SQL queries)
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Web Scraping & APIs:** `requests`, `beautifulsoup4`, `lxml`
* **File Handling:** `openpyxl` (for .xlsx), `pyarrow`, `fastparquet` (for .parquet), `s3fs` (for S3 access)
* **Image Processing:** `Pillow`
* **Note:** For image analysis, rely on your vision capabilities to analyze images provided in the conversation. Use code only for data processing tasks.

---

**CRITICAL OUTPUT FORMATTING RULES (for the Analysis script in Step 2):**

1. Your final script must print a single, raw JSON object or array to standard output.
2. Do not add any descriptive text or keys other than those explicitly requested.
3. Study this example carefully:
   - USER'S QUESTION: "Return a JSON object with keys 'rows' and 'avg_price'. 1. How many rows? 2. What is the average price?"
   - **CORRECT SCRIPT (after inspection reveals 'price' column):**
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
4. Output format can be changed as requested in the questions file / instructions.
5. For tasks involving both images and data, combine insights from both sources in your final JSON output.

---

**Your Instructions Summary:**

1. I will give you a question and file(s) (data files and/or images).
2. You will respond with an **inspection** (a script for data files, a text analysis for images).
3. I will give you the output (column names, code results, etc.).
4. You will respond with the final **analysis script** or a comprehensive analysis.
5. I will give you the output of that step.
6. You will respond with **OK**.

"""

async def run_analysis_loop(question_text: str, data_files: Dict[str, bytes], image_files: Dict[str, bytes]):
    """
    Core agent logic with support for both data files and images using the new Google GenAI SDK.
    """
    all_files = {**data_files, **image_files}
    file_list_str = ", ".join(all_files.keys()) if all_files else "None"
    
    # A list of benign warnings to ignore from stderr.
    stderr_ignore_list = [
        "Matplotlib is building the font cache",
        "DeprecationWarning",
        "UserWarning"
    ]
    
    # Safety settings using the new SDK format
    safety_settings = [
        types.SafetySetting(
            category='HARM_CATEGORY_HARASSMENT',
            threshold='BLOCK_NONE'
        ),
        types.SafetySetting(
            category='HARM_CATEGORY_HATE_SPEECH', 
            threshold='BLOCK_NONE'
        ),
        types.SafetySetting(
            category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
            threshold='BLOCK_NONE'
        ),
        types.SafetySetting(
            category='HARM_CATEGORY_DANGEROUS_CONTENT',
            threshold='BLOCK_NONE'
        ),
    ]
    
    # Create chat using the new SDK - no config parameter needed for basic chat
    chat = client.aio.chats.create(model=LLM_MODEL)
    
    # Prepare initial prompt with image context
    data_file_list = ", ".join(data_files.keys()) if data_files else "None"
    image_file_list = ", ".join(image_files.keys()) if image_files else "None"
    
    initial_prompt = f"""Here is my question:
---
{question_text}
---
Available data files: [{data_file_list}]
Available image files: [{image_file_list}]"""

    feedback = initial_prompt
    last_successful_output = None

    for i in range(MAX_ITERATIONS):
        print(f"--- Iteration {i+1} ---")
        llm_response = ""

        try:
            # Create the message config with system instruction and safety settings
            message_config = types.GenerateContentConfig(
                system_instruction=get_system_prompt(),
                safety_settings=safety_settings,
            )
            
            # For the first iteration, include images if available
            if i == 0 and image_files:
                # Create content with both text and images
                message_content = [feedback]
                for filename, image_data in image_files.items():
                    mime_type = get_image_mime_type(filename)
                    # Create a PIL Image from bytes for the new SDK
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image_data))
                    message_content.append(image)
                
                response = await chat.send_message(
                    message=message_content,
                    config=message_config
                )
            else:
                response = await chat.send_message(
                    message=feedback,
                    config=message_config
                )
            
            llm_response = response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            print(f"LLM API call failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM API call failed: {str(e)}")

        if llm_response.strip().upper() == "OK":
            print("✅ LLM signaled completion.")
            if last_successful_output:
                # Validate that the output looks like valid JSON/data
                if not last_successful_output.strip():
                    raise HTTPException(status_code=500, detail="Agent finished but produced empty output.")
                return Response(content=last_successful_output, media_type="application/json")
            raise HTTPException(status_code=500, detail="Agent finished without producing any output.")
        
        # Check if LLM is trying to end prematurely with fabricated data
        if any(phrase in llm_response.lower() for phrase in ['based on the data', 'from the analysis', 'the results show']):
            if not last_successful_output:
                print("⚠️  LLM might be fabricating results without code execution!")
                feedback = (
                    "WARNING: You seem to be providing analysis without executing code first.\n"
                    "STRICT REQUIREMENTS:\n"
                    "1. You must execute Python code to access the actual data\n"
                    "2. All results must come from real code execution\n"
                    "3. Do not make assumptions about data contents\n"
                    "4. Only provide conclusions after successful code execution\n\n"
                    "Please provide the Python code to analyze the actual data."
                )
                continue

        code = extract_python_code(llm_response)
        if not code:
            # Check if the LLM is trying to provide final results without code execution
            if any(keyword in llm_response.lower() for keyword in ['json', '{', 'result', 'answer', 'final']):
                print(f"⚠️  LLM attempting to provide results without code execution!")
                feedback = (
                    "ERROR: You provided results without executing code first.\n"
                    "MANDATORY RULES:\n"
                    "1. You MUST write and execute Python code to get actual data\n"
                    "2. NEVER fabricate, estimate, or make up any numbers\n"
                    "3. Only provide final JSON results AFTER successful code execution\n"
                    "4. If you're in the inspection phase, provide inspection code\n"
                    "5. If you're in the analysis phase, provide analysis code\n\n"
                    "Please provide the required Python code now."
                )
                continue
            
            # If no code, this might be legitimate image analysis or a text-based response
            print(f"No code found. LLM Response: {llm_response[:200]}...")
            feedback = (
                "I received your analysis. If this completes the task, respond with `OK`. "
                "If you need to write Python code for further analysis, please provide it now."
            )
            continue

        print(f"Executing code:\n{code[:350]}...")
        stdout, stderr = await execute_code(code, data_files)  # Only pass data files for execution

        # Check for real errors (ignoring benign warnings)
        is_real_error = stderr and not any(warning in stderr for warning in stderr_ignore_list)

        if is_real_error:
            print(f"Execution Error: {stderr}")
            last_error_line = stderr.strip().split('\n')[-1]
            feedback = (
                f"CRITICAL ERROR: Your code failed with error: `{last_error_line}`\n"
                f"Full error details:\n{stderr}\n\n"
                "IMPORTANT RULES:\n"
                "1. You MUST write and execute working Python code to get the actual data\n"
                "2. Do NOT make up, estimate, or fabricate any numbers or results\n"
                "3. Do NOT provide a JSON response without successful code execution\n"
                "4. Fix the error in your code and provide the corrected Python script\n"
                "5. Only respond with 'OK' after successful code execution with real results\n\n"
                "Please analyze the error carefully and provide the corrected code."
            )
        else:
            if stderr:  # This means there was a warning, but we're ignoring it
                print(f"Execution Succeeded with a benign warning: {stderr.strip()}")
            print(f"Execution Success. Output:\n{stdout[:350]}...")
            last_successful_output = stdout
            feedback = (
                "Your code ran successfully. Here is the output:\n---\n"
                f"{stdout}\n---\n"
                "If this output represents the complete and final answer in the correct format, "
                "respond ONLY with the word `OK`. If more steps are needed (e.g., this was an "
                "inspection step), provide the next block of Python code for the analysis."
            )
    
    raise HTTPException(status_code=500, detail=f"Agent could not complete the task in {MAX_ITERATIONS} iterations.")

@app.post("/api/", tags=["Data Analysis"])
async def analyze_data(request: Request):
    """
    The main API endpoint with support for both data files and images.
    """
    try:
        form = await request.form()

        if "questions.txt" not in form:
            raise HTTPException(status_code=400, detail="questions.txt is a required field.")
            
        question_text = (await form['questions.txt'].read()).decode("utf-8")
        
        # Separate data files and image files
        data_files = {}
        image_files = {}
        
        for key, file_obj in form.items():
            if key != "questions.txt":  # Exclude the question file
                file_content = await file_obj.read()
                if is_image_file(key):
                    image_files[key] = file_content
                else:
                    data_files[key] = file_content
        
        print(f"Starting analysis with a {OVERALL_TIMEOUT}-second timeout.")
        print(f"Data files: {list(data_files.keys())}")
        print(f"Image files: {list(image_files.keys())}")
        
        return await asyncio.wait_for(
            run_analysis_loop(question_text, data_files, image_files),
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