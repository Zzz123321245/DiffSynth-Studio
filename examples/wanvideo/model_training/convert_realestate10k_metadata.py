import argparse
import csv
import json


def build_parser():
    parser = argparse.ArgumentParser(description="Convert RealEstate10K annotations to Wan camera-control metadata.")
    parser.add_argument("--input_json", required=True, help="Path to RealEstate10K json annotation file.")
    parser.add_argument("--output_csv", required=True, help="Path to output csv metadata file.")
    parser.add_argument("--video_key", default="video_path", help="Key name for video path in input json.")
    parser.add_argument("--caption_key", default="caption", help="Key name for caption in input json.")
    parser.add_argument("--pose_key", default="pose_file_aligned", help="Key name for aligned pose file path in input json.")
    parser.add_argument("--width_key", default="width", help="Key name for pose/video original width in input json.")
    parser.add_argument("--height_key", default="height", help="Key name for pose/video original height in input json.")
    return parser


def main():
    args = build_parser().parse_args()
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {args.input_json}, got {type(data)}")

    fieldnames = [
        "video",
        "prompt",
        "camera_control_pose_file",
        "camera_control_pose_width",
        "camera_control_pose_height",
    ]
    rows = []
    skipped = 0
    for item in data:
        video = item.get(args.video_key)
        prompt = item.get(args.caption_key)
        pose_file = item.get(args.pose_key)
        if video is None or prompt is None or pose_file is None:
            skipped += 1
            continue
        rows.append({
            "video": video,
            "prompt": prompt,
            "camera_control_pose_file": pose_file,
            "camera_control_pose_width": int(item.get(args.width_key, 1280)),
            "camera_control_pose_height": int(item.get(args.height_key, 720)),
        })

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output_csv}. Skipped {skipped} rows.")


if __name__ == "__main__":
    main()
