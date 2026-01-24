#!/usr/bin/env python3
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO .pt to ONNX")
    parser.add_argument("--pt", default="models/best.pt", help="Path to .pt weights")
    parser.add_argument("--out", default="models/yolo26n.onnx", help="Path to output .onnx")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("ERROR: ultralytics is required. Install with: pip install ultralytics", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.pt):
        print(f"ERROR: weights not found: {args.pt}", file=sys.stderr)
        return 1

    model = YOLO(args.pt)
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        nms=False,
        dynamic=False,
        half=False,
    )

    # ultralytics writes рядом с .pt по умолчанию, переносим/переименовываем в нужное место
    exported = os.path.splitext(args.pt)[0] + ".onnx"
    if os.path.abspath(exported) != os.path.abspath(args.out):
        os.replace(exported, args.out)

    print(f"OK: exported to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
