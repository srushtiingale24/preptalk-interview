# backend/app/main.py
import os
import shutil
import uuid
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Use relative import so Python can find model when app is run as a package
from .model import load_questions_for_role, speech_to_text_whisper, evaluate_answer

app = FastAPI(title="AI Interviewer Backend")

# adjust origins to your dev / production origins
origins = [
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],  # during dev we allow all; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
QUESTIONS_DIR = BASE_DIR / "data" / "questions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)


class GetQuestionRequest(BaseModel):
    role: str


@app.post("/api/get-question")
async def get_question(req: GetQuestionRequest):
    """
    Returns a random question row for a role.
    Response JSON:
      { "question_id": "...", "question": "...", "expected": "...", "keywords": [...] }
    """
    try:
        df = load_questions_for_role(req.role, questions_dir=str(QUESTIONS_DIR))
    except Exception as e:
        return {"error": str(e)}

    # pick one random row
    row = df.sample(1).iloc[0]
    q_text = str(row.get("Questions", "")).strip()
    expected = str(row.get("Expected Answers", "")).strip()
    keywords_raw = row.get("Keywords", "")

    # Normalise keywords into an array
    keywords = []
    try:
        if pd.isna(keywords_raw) or keywords_raw is None:
            keywords = []
        elif isinstance(keywords_raw, (list, tuple)):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            # split on commas or semicolons
            keywords = [k.strip() for k in str(keywords_raw).replace(";", ",").split(",") if k.strip()]
    except Exception:
        keywords = []

    question_id = str(uuid.uuid4())
    return {"question_id": question_id, "question": q_text, "expected": expected, "keywords": keywords}


# helper: convert webm/ogg to wav using ffmpeg
def convert_to_wav(src_path: str, dst_path: str):
    """
    Convert any audio file ffmpeg can read to 16kHz mono WAV.
    Requires `ffmpeg` in PATH.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but was not found on PATH.")
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", dst_path]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@app.post("/api/analyze-answer")
async def analyze_answer(
    audio: UploadFile = File(...),
    question: str = Form(...),
    role: str = Form(...),
    question_id: Optional[str] = Form(None),
):
    """
    Accepts multipart FormData:
      - audio: file
      - question: text
      - role: role id
      - question_id: optional
    Returns evaluation JSON (as created by evaluate_answer)
    """
    temp_name = str(uuid.uuid4())
    src_path = UPLOAD_DIR / f"{temp_name}_{audio.filename}"
    # Save incoming file
    try:
        with open(src_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        return {"error": f"Failed to save uploaded file: {e}"}

    # Convert to wav for model consistency
    wav_path = UPLOAD_DIR / f"{temp_name}.wav"
    try:
        convert_to_wav(str(src_path), str(wav_path))
    except Exception as ex:
        # If conversion fails, fallback to original file (may or may not work)
        # Log the exception (stdout)
        print("ffmpeg conversion error:", ex)
        wav_path = src_path

    # Run ASR (whisper) to get transcript
    try:
        transcript = speech_to_text_whisper(str(wav_path))
    except Exception as e:
        transcript = ""
        print("ASR error:", e)

    # Try to load expected answer and keywords from CSV
    expected = ""
    keywords = []
    try:
        df = load_questions_for_role(role, questions_dir=str(QUESTIONS_DIR))
        # find exact matching question text (best-effort)
        matched = df[df["Questions"].astype(str).str.strip() == str(question).strip()]
        if len(matched) >= 1:
            row = matched.iloc[0]
            expected = str(row.get("Expected Answers", "")).strip()
            kw_raw = row.get("Keywords", "")
            if not (kw_raw is None or (isinstance(kw_raw, float) and pd.isna(kw_raw))):
                if isinstance(kw_raw, (list, tuple)):
                    keywords = [str(k).strip() for k in kw_raw if str(k).strip()]
                else:
                    keywords = [k.strip() for k in str(kw_raw).replace(";", ",").split(",") if k.strip()]
    except Exception:
        # ignore; expected/keywords remain default
        pass

    # Run evaluation using the model
    try:
        eval_result = evaluate_answer(question, expected, keywords, str(wav_path), transcript)
        # attach extra fields
        eval_result["transcript"] = transcript
        eval_result["question_id"] = question_id
        return eval_result
    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}
    finally:
        # cleanup saved files to avoid disk growth (best-effort)
        try:
            if src_path.exists():
                src_path.unlink()
        except Exception:
            pass
        try:
            if wav_path.exists() and wav_path != src_path:
                wav_path.unlink()
        except Exception:
            pass
