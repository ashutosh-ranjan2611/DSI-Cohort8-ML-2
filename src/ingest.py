"""
Download, extract, and validate the UCI Bank Marketing dataset.

Handles the nested ZIP structure:
  bank+marketing.zip → bank-additional.zip → bank-additional-full.csv
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger(__name__)

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_DIR = Path("data/raw")
TARGET_CSV = "bank-additional-full.csv"
INNER_ZIP = "bank-additional.zip"
EXPECTED_SHAPE = (41188, 21)
TARGET_COL = "y"


def download_and_extract(url: str = DATA_URL, dest_dir: Path = RAW_DIR) -> Path:
    """Download and extract CSV from UCI's nested ZIP archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / TARGET_CSV

    if csv_path.exists():
        logger.info("Dataset already exists at %s", csv_path)
        return csv_path

    zip_path = dest_dir / "bank-marketing.zip"
    logger.info("Downloading from %s ...", url)
    urlretrieve(url, zip_path)

    # Extract: outer ZIP may contain inner ZIPs or the CSV directly
    found = False
    with zipfile.ZipFile(zip_path, "r") as outer:
        # Try direct extraction first
        for member in outer.namelist():
            if member.endswith(TARGET_CSV):
                with outer.open(member) as src, open(csv_path, "wb") as dst:
                    dst.write(src.read())
                found = True
                break

        # If not found, look inside nested ZIPs
        if not found:
            for member in outer.namelist():
                if member.endswith(".zip"):
                    inner_bytes = outer.read(member)
                    try:
                        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                            for inner_member in inner.namelist():
                                if inner_member.endswith(TARGET_CSV):
                                    with (
                                        inner.open(inner_member) as src,
                                        open(csv_path, "wb") as dst,
                                    ):
                                        dst.write(src.read())
                                    found = True
                                    break
                    except zipfile.BadZipFile:
                        continue
                if found:
                    break

    zip_path.unlink()

    if not found:
        raise FileNotFoundError(
            f"'{TARGET_CSV}' not found in archive. "
            f"Download manually from https://archive.ics.uci.edu/dataset/222/bank+marketing"
        )

    logger.info("Extracted to %s", csv_path)
    return csv_path


def load_raw_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load and validate the raw dataset. Auto-downloads if missing."""
    if filepath is None:
        filepath = RAW_DIR / TARGET_CSV
    if not filepath.exists():
        filepath = download_and_extract()

    df = pd.read_csv(filepath, sep=";")

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected {EXPECTED_SHAPE}, got {df.shape}")

    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})
    if df[TARGET_COL].isnull().any():
        raise ValueError("Unmapped target values found")

    logger.info(
        "Loaded %d records | Positive rate: %.1f%%",
        len(df),
        df[TARGET_COL].mean() * 100,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_and_extract()
    df = load_raw_data()
    print(f"✅ {df.shape[0]} records, {df.shape[1]} columns")
    print(f"Target: {df[TARGET_COL].value_counts().to_dict()}")
