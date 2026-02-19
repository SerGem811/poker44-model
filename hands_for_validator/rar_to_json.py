from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.parser_bridge import extract_header, parse_hand, split_hands


def extract_rar(rar_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(rar_path), "-C", str(extract_dir)], check=True)


def parse_hands(source_dir: Path) -> tuple[list[dict], dict[str, int]]:
    files = sorted(source_dir.rglob("*.txt"))
    stats = {
        "files": len(files),
        "raw_hands": 0,
        "parsed_hands": 0,
        "failed_parse": 0,
    }
    results: list[dict] = []

    for idx, file_path in enumerate(files, start=1):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        raw_hands = split_hands(text)
        stats["raw_hands"] += len(raw_hands)

        for raw in raw_hands:
            header = extract_header(raw, str(file_path.relative_to(source_dir)))
            try:
                parsed = parse_hand(raw)
            except Exception:
                parsed = None

            if not parsed:
                stats["failed_parse"] += 1
                continue

            results.append(
                {
                    "external_hand_id": header.get("external_hand_id"),
                    "table_name": header.get("table_name"),
                    "played_at_raw": header.get("played_at_raw"),
                    "played_tz": header.get("played_tz"),
                    "source_file": header.get("source_file"),
                    "data": parsed,
                }
            )
            stats["parsed_hands"] += 1

        if idx % 100 == 0:
            print(f"Procesados {idx}/{len(files)} archivos...")

    return results, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae un .rar de manos de poker y lo convierte a JSON.")
    parser.add_argument("--rar-path", default="../poker_hands.rar", help="Ruta al archivo .rar")
    parser.add_argument("--extract-dir", default="../extracted_rar", help="Directorio para descomprimir")
    parser.add_argument(
        "--source-dir",
        default="../extracted_rar/poker_hands",
        help="Directorio con .txt descomprimidos",
    )
    parser.add_argument("--output-json", default="../data/poker_hands_from_rar.json", help="Ruta JSON de salida")
    parser.add_argument("--skip-extract", action="store_true", help="No descomprime el .rar")
    args = parser.parse_args()

    rar_path = Path(args.rar_path).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_json = Path(args.output_json).resolve()

    if not args.skip_extract:
        if not rar_path.exists():
            raise SystemExit(f"No existe el .rar: {rar_path}")
        print(f"Descomprimiendo {rar_path} -> {extract_dir}")
        extract_rar(rar_path, extract_dir)

    if not source_dir.exists():
        raise SystemExit(f"No existe el directorio fuente: {source_dir}")

    print(f"Parseando .txt desde {source_dir}")
    hands, stats = parse_hands(source_dir)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_rar": str(rar_path),
        "source_dir": str(source_dir),
        "stats": stats,
        "hands": hands,
    }

    output_json.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    print(f"JSON generado en {output_json}")
    print(stats)


if __name__ == "__main__":
    main()
