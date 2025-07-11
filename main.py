# --- 1.Necessary imports ---
import os
import json
import io
import google.generativeai as genai
import PIL.Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import fitz
from dotenv import load_dotenv
load_dotenv()

# --- 2. Configuration ---

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print("FATAL ERROR: Could not configure Google AI. Is GOOGLE_API_KEY set correctly?")
    raise e

# --- 3. Initialize the FastAPI application ---
app = FastAPI(
    title="German Letter Analyzer API",
    description="An API that uses Google Gemini to analyze scanned German letters.",
    version="1.0.0"
)

# --- 4. The Core Logic: The AI System Prompt ---
SYSTEM_PROMPT = """
You are an expert German administrative assistant. Your task is to analyze the content of a letter provided as an image or text.
Analyze the letter and extract key information.
You MUST respond ONLY with a valid JSON object. Do not include any text before or after the JSON.

The JSON object must have the following structure:
{
  "category": "One of: DEADLINE, FINANCIAL, INFO",
  "summary_german": "A concise 2-3 sentence summary of the letter in German.",
  "deadline_date": "If a specific deadline or payment due date is mentioned (e.g., 'fällig am', 'zahlbar bis zum 31.05.2024'), provide it in YYYY-MM-DDTHH:MM:SS format. Use T21:00:00 for the time (end of day). If no date, this must be null.",
  "deadline_subject": "A short subject for a calendar event, e.g., 'Antrag für XYZ einreichen' or for bills 'Payment for ABC Corp'. If no action, this must be null.",
  "payment_amount": "The numeric value of any payment required. Otherwise, null.",
  "payment_currency": "The currency symbol or code (e.g., 'EUR'). Otherwise, null.",
  "payment_recipient": "Who the payment is for. Otherwise, null.",
  "full_analysis_log": "A detailed breakdown of the letter's purpose, key points, and extracted entities."
}

---
**VERY IMPORTANT: CATEGORIZATION RULES**

Follow these rules in order. Stop at the first rule that matches.

1.  **FINANCIAL Check:** First, scan the document for strong financial keywords like "Rechnung", "Mahnung", "Betrag", "Kostenaufstellung", "fällig", "zu zahlen". 
    - If you find any of these, you **MUST** set the category to "FINANCIAL".
    - Even if there is a date, if it is primarily a bill, the category is "FINANCIAL". Do not proceed to the next rule.

2.  **DEADLINE Check:** If, and only if, the document is NOT financial, then check for non-payment deadlines. Look for keywords like "Antrag bis", "spätestens bis", "fristgerecht einreichen", "Antwort bis".
    - If you find these, set the category to "DEADLINE".

3.  **INFO Fallback:** If neither of the above rules apply, the category is "INFO".
---
"""
# --- 5. The API Endpoint ---
@app.post("/analyze-letter/")
async def analyze_letter(file: UploadFile = File(...)):
    """
    Receives an image (PNG, JPG) or PDF file.
    If it's a PDF, it converts EVERY page to an image.
    Then analyzes all images with Gemini 1.5 flash and returns a single structured JSON response.
    """
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        file_content = await file.read()
        
        model_content = [SYSTEM_PROMPT]

        if file.content_type == "application/pdf":
            print(f"PDF detected. Converting all pages to images...")
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            for i, page in enumerate(pdf_document):
                print(f"  - Processing page {i+1}/{len(pdf_document)}")
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                img = PIL.Image.open(io.BytesIO(img_bytes))
                model_content.append(img)
            
            print("PDF conversion successful.")

        else: # If it's a single JPG or PNG
            img = PIL.Image.open(io.BytesIO(file_content))
            model_content.append(img)
            
        # The Gemini call now sends the list of all images
        model = genai.GenerativeModel('gemini-1.5-flash')
        print(f"Sending request with {len(model_content) - 1} image(s) to Google Gemini API...")
        response = model.generate_content(
            model_content,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        
        print("Received response from API.")
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        analysis_result = json.loads(response_text)
        print("Successfully parsed JSON response.")
        return JSONResponse(content=analysis_result)

    
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON from model response. Response was:\n{response.text}")
        raise HTTPException(status_code=500, detail="The AI model returned an invalid format.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- 6. (Optional) A simple root endpoint for testing ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "German Letter Analyzer API is running."}

# This part allows running the app directly with 'python main.py'
# But for development, 'uvicorn main:app --reload' is better.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)