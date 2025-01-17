# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import Any

import pytest

from paddleocr import PaddleOCR

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

IMAGE_PATHS_OCR = [str(test_file_dir / "idcard.png")]


@pytest.fixture(params=["korean"])
def ocr_engine(request: Any) -> PaddleOCR:
    return PaddleOCR(lang=request.param, use_triton=True)


def test_ocr_initialization(ocr_engine: PaddleOCR) -> None:
    assert ocr_engine is not None


@pytest.mark.parametrize("image_path", IMAGE_PATHS_OCR)
def test_ocr_function(ocr_engine: PaddleOCR, image_path: str) -> None:
    """
    Test PaddleOCR OCR functionality with different images.

    Args:
        ocr_engine: An instance of PaddleOCR.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.ocr(image_path)

    for line in result:
        for box, (txt, score) in line:
            print(f"box: {box}, txt: {txt}, score: {score}")
    assert result is not None
    assert isinstance(result, list)


def test_triton_model_info():
    import tritonclient.grpc as grpcclient

    triton_client = grpcclient.InferenceServerClient(
        url="localhost:8001",
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None)

    model_info = triton_client.get_model_config("ppocr_det")
    # 모델 정보 확인
    print("--------------------------------")
    print("Model Info", end=": ")
    print(triton_client.get_model_config("ppocr_det"))