import argparse

from ov_paddleocr_vl import PaddleOCR_VL_OV


def parse_args():
    """Parse CLI arguments for PaddleOCR_VL_OV export."""
    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR-VL model to OpenVINO format."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./PaddleOCR-VL",
        help="Path to the original PaddleOCR-VL pretrained model.",
    )
    parser.add_argument(
        "--ov_model_path",
        type=str,
        default="./ov_paddleocr_vl_model",
        help="Destination path for the exported OpenVINO model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device used for OpenVINO compilation.",
    )
    parser.add_argument(
        "--llm_int4_compress",
        action="store_true",
        help="Enable INT4 compression for the LLM part.",
    )
    parser.add_argument(
        "--vision_int8_quant",
        action="store_true",
        help="Enable INT8 quantization for the vision encoder.",
    )
    parser.add_argument(
        "--llm_int8_quant",
        action="store_true",
        help="Enable INT8 quantization for the LLM part if supported.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paddleocr_vl_ov = PaddleOCR_VL_OV(
        pretrained_model_path=args.pretrained_model_path,
        ov_model_path=args.ov_model_path,
        device=args.device,
        llm_int4_compress=args.llm_int4_compress,
        vision_int8_quant=args.vision_int8_quant,
        llm_int8_quant=args.llm_int8_quant,
    )
    paddleocr_vl_ov.export_vision_to_ov()


if __name__ == "__main__":
    main()
