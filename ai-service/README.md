# AI Image Authenticity Service

Production-ready FastAPI microservice for detecting whether an image is **real** or **AI-generated**. Converted from the original Streamlit application with **zero changes** to prediction logic, preprocessing, or trained model weights.

Supports three classifiers:
- **CNN** — custom convolutional neural network
- **EfficientNetB3** — general-purpose real vs. AI detector
- **EfficientNet Art** — fine-tuned for art-style images

---

## Project Structure

```
ai-service/
├── app/
│   ├── main.py              # FastAPI app, middleware, startup
│   └── config.py            # Settings, paths, constants
├── routes/
│   └── verify.py            # GET /health, POST /verify
├── services/
│   ├── model_loader.py      # Singleton model loading (startup only)
│   ├── predictor.py         # Preprocess → predict → format
│   ├── metadata.py          # Model name/version lookup
│   └── authenticity.py      # Raw score → REAL/AI interpretation
├── utils/
│   ├── preprocessing.py     # Image preprocessing pipelines
│   └── image_utils.py       # Upload validation and streaming
├── models/
│   ├── efficientnetb3_binary_classifier_8.h5
│   ├── EfficientNet_fine_tune_art_model.h5
│   └── model_weights.weights.h5
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Run Locally

### Prerequisites

- **Python 3.11** (required — TensorFlow 2.18 does not support Python 3.12+ or 3.14)
- Model weight files in `models/` (already included if copied from the parent project)

> If only Python 3.14 is installed locally, use Docker instead (see below) or install Python 3.11 from [python.org](https://www.python.org/downloads/).

### 1. Create and activate a virtual environment

```powershell
cd ai-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

| URL | Description |
|-----|-------------|
| http://localhost:8000/health | Health check |
| http://localhost:8000/verify | Image verification |
| http://localhost:8000/docs | Swagger UI (auto-generated) |
| http://localhost:8000/redoc | ReDoc (auto-generated) |

> **Note:** First startup takes 30–90 seconds while TensorFlow loads all three models into memory.

---

## Docker Commands

### Build and run with Docker Compose (recommended)

```powershell
cd ai-service
docker compose up --build
```

### Run in detached mode

```powershell
docker compose up --build -d
```

### View logs

```powershell
docker compose logs -f ai-service
```

### Stop the service

```powershell
docker compose down
```

### Build image only

```powershell
docker build -t ai-image-detector:latest .
```

### Run container manually

```powershell
docker run -p 8000:8000 --name ai-detector ai-image-detector:latest
```

---

## API Reference

### `GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

---

### `POST /verify`

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `image` | file | Yes | JPG, JPEG, PNG |
| `model` | string | Yes | `cnn`, `efficientnet`, `efficientnet-art` |

**Success Response (200):**
```json
{
  "prediction": "REAL",
  "confidence": 98.4,
  "real_probability": 98.4,
  "ai_probability": 1.6,
  "model": "EfficientNetB3",
  "model_version": "1.0.0",
  "processing_time_ms": 341,
  "timestamp": "2026-07-10T10:00:00.000000+00:00",
  "success": true
}
```

**Error Responses:**

| Code | Cause |
|------|-------|
| 400 | Invalid file type, empty file, or file too large |
| 404 | Unknown model identifier |
| 500 | Prediction or internal server error |

---

## API Testing Examples

### cURL — Health check

```bash
curl http://localhost:8000/health
```

### cURL — Verify with EfficientNet

```bash
curl -X POST http://localhost:8000/verify \
  -F "image=@/path/to/your/image.jpg" \
  -F "model=efficientnet"
```

### cURL — Verify with CNN

```bash
curl -X POST http://localhost:8000/verify \
  -F "image=@/path/to/your/image.png" \
  -F "model=cnn"
```

### cURL — Verify with EfficientNet Art

```bash
curl -X POST http://localhost:8000/verify \
  -F "image=@/path/to/artwork.jpg" \
  -F "model=efficientnet-art"
```

### PowerShell — Verify

```powershell
$uri = "http://localhost:8000/verify"
$form = @{
    image = Get-Item -Path "C:\path\to\image.jpg"
    model = "efficientnet"
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

### Python requests

```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/verify",
        files={"image": ("image.jpg", f, "image/jpeg")},
        data={"model": "efficientnet"},
    )

print(response.json())
```

---

## Node.js / Express Integration

### Axios example

```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

async function verifyImage(imagePath, model = "efficientnet") {
  const form = new FormData();
  form.append("image", fs.createReadStream(imagePath));
  form.append("model", model);

  try {
    const response = await axios.post(
      "http://localhost:8000/verify",
      form,
      {
        headers: {
          ...form.getHeaders(),
          "X-Request-ID": `express-${Date.now()}`,
        },
        timeout: 30000,
      }
    );

    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(
        `AI service error ${error.response.status}: ${JSON.stringify(error.response.data)}`
      );
    }
    throw error;
  }
}

// Usage
verifyImage("./uploads/photo.jpg", "efficientnet")
  .then((result) => {
    console.log(`Prediction: ${result.prediction}`);
    console.log(`Confidence: ${result.confidence}%`);
  })
  .catch(console.error);
```

### Express route example

```javascript
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const upload = multer({ storage: multer.memoryStorage() });
const router = express.Router();

router.post("/api/verify-image", upload.single("image"), async (req, res) => {
  const form = new FormData();
  form.append("image", req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype,
  });
  form.append("model", req.body.model || "efficientnet");

  const { data } = await axios.post(
    process.env.AI_SERVICE_URL || "http://localhost:8000/verify",
    form,
    { headers: form.getHeaders() }
  );

  res.json(data);
});
```

---

## Production Recommendations

1. **Single worker per container** — TensorFlow models are loaded in-process. Use `--workers 1` (already set in Dockerfile). Scale horizontally with multiple containers behind a load balancer.

2. **Resource allocation** — Allocate at least **2 GB RAM** per instance (4 GB recommended). Model loading peaks memory during startup.

3. **Startup probe** — Configure a `start_period` of 120s in your orchestrator (Kubernetes/Docker) to allow model loading before marking the pod healthy.

4. **Request timeouts** — Set client timeouts to 30s. First inference after startup may be slower due to TensorFlow graph warm-up.

5. **Environment variables** — Configure via env vars:
   - `LOG_LEVEL` — `DEBUG`, `INFO`, `WARNING`
   - `MAX_UPLOAD_SIZE_BYTES` — default 10 MB
   - `PORT` / `HOST` — bind address

6. **Security**
   - Place the service behind an API gateway or reverse proxy (nginx, Traefik).
   - Restrict CORS `allow_origins` in `app/main.py` to your Express backend domain.
   - Do not expose port 8000 publicly; keep it on an internal network.

7. **Monitoring**
   - Use `GET /health` for liveness probes.
   - Forward `X-Request-ID` from your Express backend for distributed tracing.
   - Monitor `processing_time_ms` in responses for latency SLAs.

8. **GPU inference** — For higher throughput, replace `tensorflow==2.18.0` with `tensorflow[and-cuda]` and use a GPU-enabled base image. No code changes required.

9. **Model immutability** — Model weights are mounted read-only in Docker Compose. Never modify files in `models/`.

10. **Logging** — Logs include request ID, timestamp, selected model, prediction time, and errors. Pipe to your centralized logging stack (ELK, Datadog, CloudWatch).

---

## Prediction Logic (Preserved)

| Model | Input Size | Normalization | Threshold |
|-------|-----------|---------------|-----------|
| CNN | 256×256 | ÷ 255 | score < 0.5 → AI |
| EfficientNetB3 | 300×300 | none | score < 0.5 → AI |
| EfficientNet Art | 224×224 | none | score < 0.5 → AI |

The sigmoid output represents the **probability the image is REAL**. No models were retrained or modified.
