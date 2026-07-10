"""
API route handlers for health checks and image verification.
"""

import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from services.metadata import validate_model_identifier
from services.predictor import get_predictor
from utils.image_utils import read_upload_stream, validate_image_upload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Verification"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """
    Liveness probe endpoint.

    Returns a simple healthy status for load balancers and orchestrators.
    """
    return {"status": "healthy"}


@router.post("/verify")
async def verify_image(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPG, JPEG, or PNG)"),
    model: str = Form(
        ...,
        description="Model to use: cnn | efficientnet | efficientnet-art",
    ),
) -> dict:
    """
    Verify whether an uploaded image is real or AI-generated.

    Accepts multipart/form-data with:
      - `image`: the image file
      - `model`: one of cnn, efficientnet, efficientnet-art

    Returns structured JSON with prediction, confidence, and timing metadata.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(
        "Verify request started request_id=%s model=%s filename=%s",
        request_id,
        model,
        image.filename,
    )

    # --- Input validation ---
    validate_image_upload(image)
    model_id = validate_model_identifier(model)

    try:
        # Read upload in chunks (avoids loading entire file at once upfront)
        image_buffer = await read_upload_stream(image)

        # Run prediction using cached in-memory models
        predictor = get_predictor()
        result = predictor.predict(image_buffer, model_id)

        logger.info(
            "Verify request succeeded request_id=%s model=%s prediction=%s "
            "confidence=%.1f processing_time_ms=%d",
            request_id,
            model_id,
            result.prediction,
            result.confidence,
            result.processing_time_ms,
        )

        return result.to_dict()

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Prediction error request_id=%s model=%s error=%s",
            request_id,
            model_id,
            str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

    finally:
        await image.close()
