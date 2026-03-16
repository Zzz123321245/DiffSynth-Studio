import argparse
import csv
import json
import os
from pathlib import Path

import imageio


def parse_args():
    parser = argparse.ArgumentParser(description="Filter metadata by video readability via ffmpeg/imageio.")
    parser.add_argument("--dataset_base_path", type=str, required=True, help="Dataset base path used to resolve relative video paths.")
    parser.add_argument("--input_metadata_path", type=str, required=True, help="Input metadata path (.json/.jsonl/.csv).")
    parser.add_argument("--output_metadata_path", type=str, default=None, help="Output metadata path. Default: add _readable suffix.")
    parser.add_argument("--video_key", type=str, default="video_path", help="Video path key in metadata.")
    parser.add_argument("--check_first_frame", action="store_true", help="Also decode first frame to validate actual frame read.")
    parser.add_argument("--bad_rows_csv", type=str, default=None, help="Optional bad-row report CSV path.")
    parser.add_argument("--progress_every", type=int, default=500, help="Print progress every N rows.")
    return parser.parse_args()


def infer_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".jsonl":
        return input_path.with_name(f"{input_path.stem}_readable.jsonl")
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}_readable.json")
    if input_path.suffix == ".csv":
        return input_path.with_name(f"{input_path.stem}_readable.csv")
    return input_path.with_name(f"{input_path.name}_readable")


def load_rows(path: Path):
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"JSON metadata must be a list of rows: {path}")
        return rows, "json"

    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, "jsonl"

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, "csv"


def write_rows(path: Path, rows, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        return

    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    fieldnames = []
    fieldset = set()
    for row in rows:
        for key in row.keys():
            if key not in fieldset:
                fieldset.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    dataset_base = Path(args.dataset_base_path)
    input_path = Path(args.input_metadata_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_path}")

    output_path = Path(args.output_metadata_path) if args.output_metadata_path else infer_output_path(input_path)
    bad_rows_csv = Path(args.bad_rows_csv) if args.bad_rows_csv else output_path.with_name(f"{output_path.stem}_bad.csv")

    rows, fmt = load_rows(input_path)
    total = len(rows)

    good_rows = []
    bad_rows = []

    for idx, row in enumerate(rows):
        rel_path = row.get(args.video_key)
        if rel_path is None:
            bad_rows.append({
                "row_index": idx,
                "video_rel_path": "",
                "video_abs_path": "",
                "reason": f"missing_key:{args.video_key}",
                "error": "",
            })
            continue

        rel_path = str(rel_path)
        if rel_path.strip() == "":
            bad_rows.append({
                "row_index": idx,
                "video_rel_path": rel_path,
                "video_abs_path": "",
                "reason": "empty_path",
                "error": "",
            })
            continue

        abs_path = Path(rel_path) if os.path.isabs(rel_path) else (dataset_base / rel_path)
        if not abs_path.exists():
            bad_rows.append({
                "row_index": idx,
                "video_rel_path": rel_path,
                "video_abs_path": str(abs_path),
                "reason": "file_not_found",
                "error": "",
            })
            continue

        if abs_path.stat().st_size <= 0:
            bad_rows.append({
                "row_index": idx,
                "video_rel_path": rel_path,
                "video_abs_path": str(abs_path),
                "reason": "file_empty",
                "error": "",
            })
            continue

        reader = None
        try:
            reader = imageio.get_reader(str(abs_path))
            _ = reader.get_meta_data()
            if args.check_first_frame:
                _ = reader.get_data(0)
            good_rows.append(row)
        except Exception as e:
            bad_rows.append({
                "row_index": idx,
                "video_rel_path": rel_path,
                "video_abs_path": str(abs_path),
                "reason": "ffmpeg_read_error",
                "error": str(e).replace("\n", " ")[:1000],
            })
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            print(f"Processed {idx + 1}/{total} rows | kept={len(good_rows)} bad={len(bad_rows)}")

    write_rows(output_path, good_rows, fmt)

    bad_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    with bad_rows_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["row_index", "video_rel_path", "video_abs_path", "reason", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in bad_rows:
            writer.writerow(item)

    print("==== Filtering done ====")
    print(f"Input rows: {total}")
    print(f"Kept rows: {len(good_rows)}")
    print(f"Dropped rows: {len(bad_rows)}")
    print(f"Output metadata: {output_path}")
    print(f"Bad rows report: {bad_rows_csv}")


if __name__ == "__main__":
    main()
