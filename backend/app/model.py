# backend/app/model.py
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from transformers import AutoTokenizer, AutoModel  # optional if you later use real embeddings
from faster_whisper import WhisperModel
from nltk import word_tokenize, pos_tag
from nltk.corpus import words
import nltk

# ensure nltk packages
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class SemanticScorer:
    def __init__(self):
        
        self.model = None

    def embed(self, text):
        vec = np.zeros(384, dtype=np.float32)
        h = abs(hash(text))
        np.random.seed(h % (2**32))
        vec[:] = np.random.rand(384)
        return torch.tensor(vec)

    def similarity(self, text1, text2):
        if not text1 or not text2:
            return 0.0
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        cos = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return max(0.0, min(1.0, cos))



def grammar_score(text):
    tokens = word_tokenize(text or "")
    if not tokens:
        return 0.0
    tagged = pos_tag(tokens)
    try:
        valid_words = sum(1 for w, t in tagged if w.lower() in words.words())
        return round((valid_words / len(tokens)) * 10, 2)
    except Exception:
        return 0.0



def analyze_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        return dict(mean_pitch=0.0, fluency=0.0, confidence=0.0, tone=0.0, duration=0.0)

    pitch, mag = librosa.piptrack(y=y, sr=sr)
    mean_pitch = float(np.mean(pitch[pitch > 0])) if np.any(pitch > 0) else 0.0
    rms = float(np.mean(librosa.feature.rms(y=y)))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pauses = int(np.sum(onset_env < 0.01)) if onset_env is not None else 0
    fluency = max(0, 10 - pauses / 10)
    confidence = float(np.clip(rms * 100, 0, 10))
    tone_score = float(np.clip((fluency + confidence) / 2, 0, 10))
    duration = float(librosa.get_duration(y=y, sr=sr))
    return dict(mean_pitch=mean_pitch, fluency=fluency, confidence=confidence, tone=tone_score, duration=duration)



_whisper_model = None
def get_whisper_model(device="cpu", model_size="base"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(model_size, device=device)
    return _whisper_model

def speech_to_text_whisper(audio_file):
    model = get_whisper_model(device="cpu", model_size="base")
    segments, info = model.transcribe(audio_file, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    return text.strip()



def evaluate_answer(question, expected, keywords, audio_file_path, text_response):
    scorer = SemanticScorer()
    semantic = scorer.similarity(text_response, expected) * 10
    grammar = grammar_score(text_response)
    keyword_hits = 0
    if keywords:
        keyword_hits = sum(1 for k in keywords if k and k.lower() in (text_response or "").lower())
    keyword_score = (keyword_hits / len(keywords)) * 10 if keywords else 0.0
    audio_feats = analyze_audio_features(audio_file_path)

    technical = (semantic + keyword_score) / 2
    communication = grammar
    confidence = audio_feats.get("confidence", 0)
    soft_skills = (audio_feats.get("tone", 0) + communication) / 2
    overall = float(np.mean([technical, communication, confidence, soft_skills]))

    feedback = {
        "technical": round(float(technical), 2),
        "communication": round(float(communication), 2),
        "confidence": round(float(confidence), 2),
        "soft_skills": round(float(soft_skills), 2),
        "overall": round(float(overall), 2),
        "semantic_similarity": round(float(semantic), 2),
        "keyword_score": round(float(keyword_score), 2),
        "grammar_score": round(float(grammar), 2),
        "tone_score": round(float(audio_feats.get("tone", 0)), 2),
        "duration_seconds": round(float(audio_feats.get("duration", 0)), 2),
    }
    return feedback



def _normalize_column_names(columns):
    """
    Map common CSV column name variants to canonical names:
      - Questions
      - Expected Answers
      - Keywords
    Returns a dict mapping original_name -> canonical_name when possible.
    """
    mapping = {}
    canonical = {
        "questions": "Questions",
        "question": "Questions",
        "ques_no": "Questions",        
        "expected answers": "Expected Answers",
        "expected_answer": "Expected Answers",
        "expected answers": "Expected Answers",
        "expected": "Expected Answers",
        "answer": "Expected Answers",
        "keywords": "Keywords",
        "key words": "Keywords",
        "tags": "Keywords",
        "tags_list": "Keywords",
    }
    for col in columns:
        key = col.strip().lower().replace("-", " ").replace("_", " ")
        if key in canonical:
            mapping[col] = canonical[key]
    return mapping


def load_questions_for_role(role, questions_dir="data/questions"):
    """
    Load CSV for role and normalize column names so the rest of the code can rely on:
      - 'Questions' (string)
      - 'Expected Answers' (string)
      - 'Keywords' (list or comma string)
    The function accepts many common variants like `question`, `expected_answer`, `keywords`, `tags`, etc.
    """
    
    filename = os.path.join(questions_dir, f"{role}.csv")
    if not os.path.exists(filename):
        filename = os.path.join(questions_dir, f"{role.replace(' ', '_').lower()}.csv")
    if not os.path.exists(filename):
        
        candidates = []
        try:
            for f in os.listdir(questions_dir):
                if f.lower().endswith(".csv") and role.lower() in f.lower():
                    candidates.append(os.path.join(questions_dir, f))
            if candidates:
                filename = candidates[0]
        except Exception:
            pass

    if not os.path.exists(filename):
        raise FileNotFoundError(f"No CSV found for role '{role}'. Tried: {filename}")

    df = pd.read_csv(filename)

    
    col_map = _normalize_column_names(df.columns)
    if col_map:
        df = df.rename(columns=col_map)

    
    if "Questions" not in df.columns:
        lower_cols = {c.lower(): c for c in df.columns}
        for alias in ("question", "questions", "ques_no", "ques"):
            if alias in lower_cols:
                df = df.rename(columns={lower_cols[alias]: "Questions"})
                break

    
    if "Questions" not in df.columns:
        raise ValueError("CSV must contain a 'Questions' column (or a variant like 'question').")

    
    if "Expected Answers" not in df.columns:
        lower_cols = {c.lower(): c for c in df.columns}
        for alias in ("expected answers", "expected_answer", "expected", "answer"):
            if alias in lower_cols:
                df = df.rename(columns={lower_cols[alias]: "Expected Answers"})
                break
        else:
            df["Expected Answers"] = ""

    
    if "Keywords" not in df.columns:
        lower_cols = {c.lower(): c for c in df.columns}
        for alias in ("keywords", "tags", "key words", "keywords_list"):
            if alias in lower_cols:
                df = df.rename(columns={lower_cols[alias]: "Keywords"})
                break
        else:
            
            df["Keywords"] = ""

    
    df["Questions"] = df["Questions"].astype(str)
    df["Expected Answers"] = df["Expected Answers"].fillna("").astype(str)

    return df
