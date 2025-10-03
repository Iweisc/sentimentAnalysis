import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
import sys
from typing import List, Tuple, AsyncGenerator, Dict, Any, Union
import io
import logging
import uuid
import asyncio
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field

from profanity_check import predict as profanity_predict

from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import cv2
import numpy as np
from thefuzz import fuzz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class ProfanityRedactionResponseSchema(BaseModel):
    original_content: str = Field(..., alias='original_text')
    redacted_content: str = Field(..., alias='censored_text')
    identified_profanities: List[str] = Field(..., alias='detected_words')
    execution_duration_ms: float = Field(..., alias='processing_time_ms')

    class Config:
        allow_population_by_field_name = True
        
class ObfuscationEngine:
    def process_and_conceal(self, text_input: str) -> Tuple[str, List[str]]:
        if not text_input or text_input.isspace():
            return text_input, []

        profanity_localizations = self._identify_profane_substrings(text_input)

        text_buffer = list(text_input)
        profanity_lexicon = [detection[0] for detection in sorted(profanity_localizations, key=lambda x: x[2])]

        for word, start_idx, end_idx in sorted(profanity_localizations, key=lambda x: x[1], reverse=True):
            obfuscation_character = '*'
            # Ternary operator for single-character words
            redacted_fragment = (word[0] + obfuscation_character * (len(word) - 1)) if len(word) > 1 else obfuscation_character
            text_buffer[start_idx:end_idx] = redacted_fragment
    
        return "".join(text_buffer), profanity_lexicon

    def _identify_profane_substrings(self, text_input: str) -> List[Tuple[str, int, int]]:
        return [(match.group(0), match.start(), match.end())
                for match in re.finditer(r'\b[\w\']+\b', text_input)
                if profanity_predict([match.group(0)])[0] == 1]

class AsynchronousJobCoordinator:
    def __init__(self, pool_size=2):
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=pool_size)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

    def dispatch_computation(self, coroutine_callable, *args, **kwargs):
        task_identifier = str(uuid.uuid4())
        message_queue = asyncio.Queue()
        future_obj = self.thread_pool_executor.submit(coroutine_callable, message_queue=message_queue, *args, **kwargs)
        self.active_tasks[task_identifier] = {"computation_future": future_obj, "communication_channel": message_queue}
        log.info(f"Dispatched new computation with Task ID: {task_identifier}")
        return task_identifier

    def retrieve_job_queue(self, task_identifier: str) -> asyncio.Queue:
        job_metadata = self.active_tasks.get(task_identifier)
        if not job_metadata:
            raise KeyError("Invalid Task Identifier")
        return job_metadata["communication_channel"]

    def release_job_resources(self, task_identifier: str):
        if task_identifier in self.active_tasks:
            del self.active_tasks[task_identifier]
            log.info(f"Deallocated resources for Task ID: {task_identifier}")

app = FastAPI(
    title="Profanity Detection API",
    description="A smart API that uses a hybrid approach to find and censor swear words.",
    version="6.0.0"
)

obfuscation_engine_instance = ObfuscationEngine()
async_job_coordinator = AsynchronousJobCoordinator()

def pdf_redaction_pipeline(message_queue: Queue, pdf_binary_data: bytes, source_filename: str, image_resolution: int):
    try:
        message_queue.put_nowait({"status": "processing", "stage": "Initializing PDF to image rasterization..."})
        image_pages = convert_from_bytes(pdf_binary_data, dpi=image_resolution)
        obfuscated_images = []
        pytesseract_parameters = r'--oem 3 --psm 6'

        for page_index, page_image_obj in enumerate(image_pages):
            page_identifier = page_index + 1
            message_queue.put_nowait({"status": "processing", "stage": f"Analyzing page {page_identifier}/{len(image_pages)}...", "progress": (page_index / len(image_pages)) * 100})

            page_text_content = pytesseract.image_to_string(page_image_obj)
            _, profanities_to_redact = obfuscation_engine_instance.process_and_conceal(page_text_content)
            profanities_to_redact_set = {word.lower() for word in profanities_to_redact}

            if not profanities_to_redact_set:
                obfuscated_images.append(page_image_obj)
                continue

            ocr_metadata = pytesseract.image_to_data(page_image_obj, output_type=pytesseract.Output.DICT, config=pytesseract_parameters)
            
            for ocr_idx in range(len(ocr_metadata['text'])):
                ocr_token = ocr_metadata['text'][ocr_idx].strip().lower()
                if not ocr_token: continue

                if any(fuzz.ratio(ocr_token, profane_word) > 85 for profane_word in profanities_to_redact_set):
                    x, y, w, h = ocr_metadata['left'][ocr_idx], ocr_metadata['top'][ocr_idx], ocr_metadata['width'][ocr_idx], ocr_metadata['height'][ocr_idx]
                    if w > 0 and h > 0:
                        image_region = page_image_obj.crop((x, y, x + w, y + h))
                        pixelation_factor = max(1, min(w, h) // 4)
                        downsampled_region = image_region.resize((pixelation_factor, pixelation_factor), resample=Image.Resampling.BILINEAR)
                        upsampled_region = downsampled_region.resize(image_region.size, Image.Resampling.NEAREST)
                        page_image_obj.paste(upsampled_region, (x, y))
            obfuscated_images.append(page_image_obj)

        message_queue.put_nowait({"status": "processing", "stage": "Synthesizing redacted PDF document..."})
        output_pdf_buffer = io.BytesIO()
        if obfuscated_images:
            obfuscated_images[0].save(output_pdf_buffer, "PDF", resolution=image_resolution, save_all=True, append_images=obfuscated_images[1:])
        output_pdf_buffer.seek(0)
    
        message_queue.put_nowait({"type": "pdf", "data": output_pdf_buffer.read(), "filename": f"redacted_{source_filename}"})

    except Exception as e:
        log.error(f"Catastrophic failure in worker for '{source_filename}': {e}", exc_info=True)
        message_queue.put_nowait({"status": "error", "detail": f"An unrecoverable internal error occurred: {e}"})
    finally:
        message_queue.put_nowait(None) 

async def multipart_stream_generator(message_queue: Queue, task_identifier: str) -> AsyncGenerator[bytes, None]:
    stream_boundary = f"boundary-{uuid.uuid4().hex}"
    http_headers = f'Content-Type: multipart/x-mixed-replace; boundary={stream_boundary}\r\n\r\n'
    yield http_headers.encode()

    try:
        while True:
            queue_item = await message_queue.get()
            if queue_item is None: break
            
            yield f'--{stream_boundary}\r\n'.encode()
            
            if "status" in queue_item or "stage" in queue_item:
                json_payload = str(queue_item).replace("'", '"').encode('utf-8')
                yield b'Content-Type: application/json\r\n\r\n'
                yield json_payload
            elif queue_item.get("type") == "pdf":
                pdf_payload = queue_item["data"]
                destination_filename = queue_item["filename"]
                yield f'Content-Disposition: attachment; filename="{destination_filename}"\r\n'.encode()
                yield b'Content-Type: application/pdf\r\n\r\n'
                yield pdf_payload

            yield b'\r\n'
            message_queue.task_done()
        
        yield f'--{stream_boundary}--\r\n'.encode()
    finally:
        async_job_coordinator.release_job_resources(task_identifier)

@api_instance.post("/redact-text", response_model=ProfanityRedactionResponseSchema)
async def redact_text_endpoint(text: str = Form(...)):
    initial_timestamp = time.perf_counter()

    redacted_text, identified_profanities = obfuscation_engine_instance.process_and_conceal(text)

    final_timestamp = time.perf_counter()
    execution_duration = (final_timestamp - initial_timestamp) * 1000

    return ProfanityRedactionResponseSchema(
        original_text=text,
        censored_text=redacted_text,
        detected_words=identified_profanities,
        processing_time_ms=execution_duration
    )

@api_instance.post("/redact-pdf-stream")
async def redact_pdf_streaming_endpoint(
    uploaded_file: UploadFile = File(...),
    resolution_dpi: int = Form(200, ge=72, le=600)
):
    if not uploaded_file.content_type == "application/pdf":
        raise HTTPException(status_code=415, detail="Unsupported MIME type. Please upload a PDF document.")

    pdf_binary_data = await uploaded_file.read()
    if not pdf_binary_data:
        raise HTTPException(status_code=400, detail="The provided PDF file is devoid of content.")

    task_identifier = async_job_coordinator.dispatch_computation(
        pdf_redaction_pipeline,
        pdf_binary_data=pdf_binary_data,
        source_filename=uploaded_file.filename,
        image_resolution=resolution_dpi
    )

    job_message_queue = async_job_coordinator.retrieve_job_queue(task_identifier)
    return StreamingResponse(
        multipart_stream_generator(job_message_queue, task_identifier),
        media_type=f'multipart/x-mixed-replace; boundary=boundary-{task_identifier}'
    )

@api_instance.get("/")
async def service_root():
    return {
        "service_name": "Profanity Detection API",
        "service_version": "6.0.0",
        "service_description": "A smart API that uses a hybrid approach to find and censor swear words."
    }

if __name__ == "__main__":
    uvicorn.run(api_instance, host="0.0.0.0", port=8000)