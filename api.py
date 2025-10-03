import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
import sys
from typing import List, Tuple
from dataclasses import dataclass
import io
import logging

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from transformers import pipeline

from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageDraw

import cv2
import numpy as np
from thefuzz import fuzz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CensorResult:
    original_text: str
    censored_text: str
    detected_words: List[str]
    confidence_scores: List[float]
    processing_time_ms: float

UNAMBIGUOUS_PROFANITIES = {
    "fuck", "shit", "piss", "bitch", "cunt", "asshole", "bastard", "dick",
    "douche", "goddamn", "hell", "motherfucker", "nigger", "pussy", "slut",
    "son of a bitch", "tits", "whore", "faggot", "damn", "crap", "bloody", "bugger", "prick", "balls"
}

class Censor:
    def __init__(self):
        self.model_loaded = False
        self.text_classifier = None
        self._load_models()

    def _load_models(self):
        try:
            model_name = "martin-ha/toxic-comment-model"
            logging.info(f"Loading model: {model_name}. This might take a moment...")
            
            self.text_classifier = pipeline(
                "text-classification",
                model=model_name,
                device=-1,
                top_k=None
            )
            
            self.model_loaded = True
            logging.info("Model loaded. Ready to censor!")

        except Exception as e:
            logging.error(f"Couldn't load the transformers model. Aborting. Error: {e}")
            sys.exit(1)

    def _get_overall_toxicity(self, text: str) -> float:
        if not text.strip(): 
            return 0.0
        try:
            results = self.text_classifier(text[:512]) 
            max_score = 0.0
            RELEVANT_LABELS = {'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'}
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                for item in results[0]:
                    if item['label'].lower() in RELEVANT_LABELS:
                        max_score = max(max_score, item['score'])
            return max_score
        except Exception as e:
            logging.error(f"Something went wrong during text classification: {e}")
            return 0.0

    def _find_toxic_words_by_contribution(self, text: str, initial_score: float) -> List[Tuple[str, float, int, int]]:
        detected = []
        words_in_text = list(re.finditer(r'\b[\w\']+\b', text))
        
        for match in words_in_text:
            word = match.group()
            if len(word) <= 2: continue
            start, end = match.start(), match.end()
            
            masked_text = text[:start] + "[MASK]" + text[end:]
            new_score = self._get_overall_toxicity(masked_text)
            contribution = initial_score - new_score
            
            if contribution > 0.15:
                detected.append((word, initial_score, start, end))
        return detected

    def _find_unambiguous_profanities(self, text: str) -> List[Tuple[str, float, int, int]]:
        detected = []
        words_in_text = list(re.finditer(r'\b[\w\']+\b', text))
        for match in words_in_text:
            if match.group().lower() in UNAMBIGUOUS_PROFANITIES:
                detected.append((match.group(), 1.0, match.start(), match.end()))
        return detected

    def detect_and_censor(self, text: str, confidence_threshold: float = 0.7, 
                         censor_method: str = "asterisk") -> CensorResult:
        start_time = time.perf_counter()
        
        if not text.strip():
            return CensorResult("", "", [], [], 0.0)
        
        word_detections = self._find_unambiguous_profanities(text)
        detected_word_set = {d[0].lower() for d in word_detections}

        overall_score = self._get_overall_toxicity(text)
        if overall_score >= confidence_threshold:
            contextual_detections = self._find_toxic_words_by_contribution(text, overall_score)
            for detection in contextual_detections:
                if detection[0].lower() not in detected_word_set:
                    word_detections.append(detection)
                    detected_word_set.add(detection[0].lower())

        detected_words = []
        confidence_scores = []
        censored = list(text)
        
        for word, score, start, end in sorted(word_detections, key=lambda x: x[2], reverse=True):
            if word not in detected_words:
                detected_words.insert(0, word)
                confidence_scores.insert(0, score)
                censored_word = self._censor_word(word, censor_method)
                censored[start:end] = censored_word
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return CensorResult(
            original_text=text,
            censored_text="".join(censored),
            detected_words=detected_words,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time
        )

    def _censor_word(self, word: str, method: str) -> str:
        if method == "asterisk":
            return word[0] + '*' * (len(word) - 1) if len(word) > 1 else '*'
        elif method == "full_asterisk":
            return '*' * len(word)
        elif method == "grawlix":
            import random
            symbols = ['#', '@', '$', '%', '&', '*']
            return ''.join(random.choice(symbols) for _ in range(len(word)))
        elif method == "block":
            return '[CENSORED]'
        else:
            return word[0] + '*' * (len(word) - 1) if len(word) > 1 else '*'

app = FastAPI(
    title="Profanity Detection API",
    description="A smart API that uses a hybrid approach to find and censor swear words.",
    version="4.2.0"
)

detector = Censor()

@app.on_event("startup")
async def startup_event():
    logging.info("API is up and running. Let the censoring begin!")

@app.post("/censor", response_model=CensorResult)
async def censor_text(
    text: str = Form(...),
    confidence_threshold: float = Form(0.7, ge=0.0, le=1.0),
    censor_method: str = Form("asterisk")
):
    """Censors a given block of text."""
    result = detector.detect_and_censor(
        text=text,
        confidence_threshold=confidence_threshold,
        censor_method=censor_method
    )
    return result

@app.post("/censor-pdf")
async def censor_pdf(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.7, ge=0.0, le=1.0)
):
    """Censors a PDF document by redacting words with black boxes."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Hey, that's not a PDF! Please upload a PDF file.")

    try:
        pdf_contents = await file.read()
        original_images = convert_from_bytes(pdf_contents)
        censored_images = []
        tesseract_config = r'--oem 3 --psm 6'

        for page_num, image in enumerate(original_images, 1):
            logging.info(f"Checking out page {page_num}...")
            
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
            processed_pil_image = Image.fromarray(binary_image)
            
            full_page_text = pytesseract.image_to_string(processed_pil_image, config=tesseract_config)
            censor_result = detector.detect_and_censor(full_page_text, confidence_threshold)
            words_to_censor = {word.lower() for word in censor_result.detected_words}

            if words_to_censor:
                logging.info(f"Found some words to censor on page {page_num}: {words_to_censor}")
            else:
                logging.info(f"Page {page_num} looks clean. Moving on.")
                censored_images.append(image)
                continue

            image_data = pytesseract.image_to_data(processed_pil_image, output_type=pytesseract.Output.DICT, config=tesseract_config)
            draw = ImageDraw.Draw(image)
            
            num_boxes = len(image_data['level'])
            for i in range(num_boxes):
                ocr_word = image_data['text'][i].strip().lower()
                if not ocr_word: continue
                
                for profane_word in words_to_censor:
                    if fuzz.ratio(ocr_word, profane_word) > 85:
                        logging.info(f"Redacting '{ocr_word}' on page {page_num} (it's a lot like '{profane_word}').")
                        x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i]
                        draw.rectangle([x, y, x + w, y + h], fill='black')
                        break
            
            censored_images.append(image)

        pdf_buffer = io.BytesIO()
        if censored_images:
            censored_images[0].save(pdf_buffer, "PDF", resolution=100.0, save_all=True, append_images=censored_images[1:])
        pdf_buffer.seek(0)
        
        headers = {'Content-Disposition': 'attachment; filename="censored_document.pdf"'}
        return StreamingResponse(pdf_buffer, headers=headers, media_type="application/pdf")

    except Exception as e:
        logging.error(f"Oh no, something went wrong with the PDF processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process the PDF. Error: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome! Send your text to /censor or a PDF to /censor-pdf."}

if __name__ == "__main__":
    print("Starting up the API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)