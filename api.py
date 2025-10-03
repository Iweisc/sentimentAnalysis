import re
import time
import sys
from typing import List, Tuple, Dict
from dataclasses import dataclass
import io
import logging

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

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

# A pre-defined list of common profanities for the OCR fallback
COMMON_PROFANITIES = {
    "fuck", "shit", "piss", "bitch", "cunt", "asshole",
    "bastard", "dick", "damn", "hell"
}

class censor:
    def __init__(self):
        self.model_loaded = False
        self._load_models()

    def _load_models(self):
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings('ignore')

            self.text_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=-1,
                top_k=None
            )

            self.token_classifier = pipeline(
                "token-classification",
                model="unitary/toxic-bert",
                device=-1,
                aggregation_strategy="max"
            )
            
            self.model_loaded = True

        except ImportError:
            logging.error("Install transformers library: pip install transformers torch")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            sys.exit(1)

    def _get_text_toxicity(self, text: str) -> Tuple[float, Dict]:
        try:
            results = self.text_classifier(text[:512])
            scores = {}
            max_score = 0.0
            
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    for item in results[0]:
                        label = item['label'].lower()
                        score = item['score']
                        scores[label] = score
                        if any(word in label for word in ['toxic', 'obscene', 'insult', 'threat']):
                            max_score = max(max_score, score)
                else:
                    label = results[0]['label'].lower()
                    score = results[0]['score']
                    scores[label] = score
                    max_score = score
            
            return max_score, scores
        except Exception as e:
            return 0.0, {}

    def _get_toxic_tokens(self, text: str) -> List[Tuple[str, float, int, int]]:
        try:
            results = self.token_classifier(text)
            detected = []
            
            for entity in results:
                word = entity['word'].strip()
                score = entity['score']
                start = entity.get('start', 0)
                end = entity.get('end', len(word))
                entity_group = entity.get('entity_group', entity.get('entity', '')).lower()
                
                if score > 0.6 and ('toxic' in entity_group or score > 0.75):
                    word_match = re.search(re.escape(word), text[max(0, start-5):end+5])
                    if word_match:
                        actual_start = max(0, start-5) + word_match.start()
                        actual_end = max(0, start-5) + word_match.end()
                        detected.append((word, score, actual_start, actual_end))
            
            return detected
        except Exception as e:
            return []

    def _ocr_fallback_detection(self, text: str) -> List[Tuple[str, float, int, int]]:
        detected = []
        words_in_text = list(re.finditer(r'\b[\w\']+\b', text))
        
        for match in words_in_text:
            word = match.group().lower()
            start, end = match.start(), match.end()
            
            for profanity in COMMON_PROFANITIES:
                if fuzz.ratio(word, profanity) > 85:
                    detected.append((match.group(), 0.9, start, end))
                    break # Move to the next word in text
        return detected

    def detect_and_censor(self, text: str, confidence_threshold: float = 0.7, 
                         censor_method: str = "asterisk") -> CensorResult:
        start_time = time.perf_counter()
        
        if not text.strip():
            return CensorResult("", "", [], [], 0.0)
        
        overall_score, _ = self._get_text_toxicity(text)
        detected_words = []
        confidence_scores = []
        censored = text
        
        if overall_score >= confidence_threshold:
            word_detections = self._get_toxic_tokens(text)
            
            if not word_detections and overall_score > 0.85:
                logging.warning("Primary model found no tokens. Using OCR fallback detection.")
                word_detections = self._ocr_fallback_detection(text)

            for word, score, start, end in reversed(sorted(word_detections, key=lambda x: x[2])):
                if word not in detected_words:
                    detected_words.insert(0, word)
                    confidence_scores.insert(0, score)
                    censored_word = self._censor_word(word, censor_method)
                    censored = censored[:start] + censored_word + censored[end:]
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return CensorResult(
            original_text=text,
            censored_text=censored,
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
            return word

app = FastAPI(
    title="Profanity Detection API",
    description="A FastAPI wrapper for a profanity detection and censorship API.",
    version="1.5.0"
)

detector = censor()

@app.on_event("startup")
async def startup_event():
    logging.info("App started successfully.")

@app.post("/censor", response_model=CensorResult)
async def censor_text(
    text: str = Form(...),
    confidence_threshold: float = Form(0.7),
    censor_method: str = Form("asterisk")
):
    result = detector.detect_and_censor(
        text=text,
        confidence_threshold=confidence_threshold,
        censor_method=censor_method
    )
    return result

@app.post("/censor-pdf")
async def censor_pdf(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.7)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        pdf_contents = await file.read()
        
        original_images = convert_from_bytes(pdf_contents)
        censored_images = []
        
        # Tesseract configuration for better OCR on document-like images
        tesseract_config = r'--oem 3 --psm 6'

        page_num = 0
        for image in original_images:
            page_num += 1
            logging.info(f"Processing Page: {page_num}")
            
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
            processed_pil_image = Image.fromarray(binary_image)

            full_page_text = pytesseract.image_to_string(processed_pil_image, config=tesseract_config)
            
            censor_result = detector.detect_and_censor(full_page_text, confidence_threshold)
            words_to_censor = {word.lower() for word in censor_result.detected_words}

            logging.info(f"[Page {page_num}] Profanity model detected: {words_to_censor if words_to_censor else 'None'}")

            if not words_to_censor:
                censored_images.append(image)
                continue

            image_data = pytesseract.image_to_data(processed_pil_image, output_type=pytesseract.Output.DICT, config=tesseract_config)
            draw = ImageDraw.Draw(image)
            
            num_boxes = len(image_data['level'])
            for i in range(num_boxes):
                word = image_data['text'][i].strip().lower()
                if not word:
                    continue
                
                should_censor = False
                matched_profane_word = ""
                for profane_word in words_to_censor:
                    similarity = fuzz.ratio(word, profane_word)
                    if similarity > 85:
                        should_censor = True
                        matched_profane_word = profane_word
                        break
                
                if should_censor:
                    logging.info(f"[Page {page_num}] Redacting OCR word '{word}' (matched with '{matched_profane_word}')")
                    x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i]
                    draw.rectangle([x, y, x + w, y + h], fill='black')
            
            censored_images.append(image)

        pdf_buffer = io.BytesIO()
        if censored_images:
            censored_images[0].save(
                pdf_buffer, "PDF", resolution=100.0, save_all=True, append_images=censored_images[1:]
            )
        pdf_buffer.seek(0)

        headers = {'Content-Disposition': 'attachment; filename="censored_document.pdf"'}
        return StreamingResponse(pdf_buffer, headers=headers, media_type="application/pdf")

    except Exception as e:
        logging.error(f"An error occurred during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")

@app.get("/")
async def root():
    return {"message": "Send POST requests to /censor with text or to /censor-pdf with a PDF file."}

if __name__ == "__main__":
    print("Starting API server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)