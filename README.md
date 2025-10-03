### Censorship API

![alt text](https://img.shields.io/badge/build-passing-brightgreen) ![alt text](https://img.shields.io/badge/python-3.11+-blue)
![alt text](https://img.shields.io/badge/license-MIT-lightgrey)



#### Its a highly performant censorship api that censors text input under ≈ sub-10ms. The api is also capable of censor large pdf files under 10s (will improve later)

## Tech stack

- Framework: FastAPI
- ML/AI: RoBERTa-large
- PDF processing: pillow, pdf2image
- OCR: pytesseract
- Fuzzy matching: thefuzz
- Web server: uvicorn

## Requirements 

   - Python 3.8+

   - Tesseract OCR Engine

   - Poppler (for PDF to image conversion)

##

-    macOS: brew install tesseract poppler

-    Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils

 -   Windows: Install via the official installers and ensure the executables are in your system's PATH.


 ### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/profanity-api.git
    cd profanity-api
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file from the imports in the script)*

4.  **Run the application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will now be running at `http://localhost:8000`.

## API Endpoints

You can access the interactive API documentation provided by FastAPI at `http://localhost:8000/docs`.

### 1. Censor Text

Censors a block of plain text.

-   **Endpoint:** `/censor`
-   **Method:** `POST`
-   **Content-Type:** `application/x-www-form-urlencoded`

**Parameters:**

| Parameter              | Type    | Default    | Description                                                               |
| ---------------------- | ------- | ---------- | ------------------------------------------------------------------------- |
| `text`                 | string  | (Required) | The text you want to censor.                                              |
| `confidence_threshold` | float   | 0.7        | The AI's confidence threshold (0.0 to 1.0) for flagging a word.           |
| `censor_method`        | string  | "asterisk" | The style of censorship to apply. See [methods](#censorship-methods) below. |

**Example Request (`curl`):**

```bash
curl -X POST "http://localhost:8000/censor" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "text=This is some fucking bullshit text that needs to be censored." \
     -d "confidence_threshold=0.7" \
     -d "censor_method=grawlix"
```

**Example Response:**

```json
{
  "original_text": "This is some fucking bullshit text that needs to be censored.",
  "censored_text": "This is some #@$%*!@ #@*!#*&@ text that needs to be censored.",
  "detected_words": [
    "fucking",
    "bullshit"
  ],
  "confidence_scores": [
    1.0,
    1.0
  ],
  "processing_time_ms": 125.43
}
```

### 2. Censor PDF

Finds and redacts profane words within a PDF document.

-   **Endpoint:** `/censor-pdf`
-   **Method:** `POST`
-   **Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter              | Type    | Default    | Description                                                             |
| ---------------------- | ------- | ---------- | ----------------------------------------------------------------------- |
| `file`                 | file    | (Required) | The PDF file to process.                                                |
| `confidence_threshold` | float   | 0.7        | The AI's confidence threshold (0.0 to 1.0) for flagging words via OCR.  |

**Example Request (`curl`):**

```bash
curl -X POST "http://localhost:8000/censor-pdf" \
     -F "file=@/path/to/your/document.pdf" \
     -F "confidence_threshold=0.8" \
     --output censored_document.pdf
```

**Response:**

The API will stream a new PDF file (`censored_document.pdf`) with the detected profanities redacted with black boxes.

## Censorship Methods

You can specify the `censor_method` in the `/censor` endpoint:

-   `asterisk` (default): Replaces all but the first letter with asterisks (e.g., `s**t`).
-   `full_asterisk`: Replaces the entire word with asterisks (e.g., `****`).
-   `grawlix`: Replaces the word with a random string of symbols (e.g., `@#$%!`).
-   `block`: Replaces the entire word with `[CENSORED]`.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/profanity-api/issues).