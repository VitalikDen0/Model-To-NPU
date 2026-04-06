#!/usr/bin/env python3
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "калибровка.json"

NEGATIVE = "(worst quality, low quality, normal quality:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, text, watermark, signature, censor, censored, bar"

CHARACTERS = [
    "Miku", "Rem", "Asuna", "Zero Two", "Saber", "Rin Tohsaka", "Mikasa", "Marin Kitagawa", "Nami", "Robin",
    "Yor Forger", "Anya Forger", "Violet Evergarden", "Megumin", "Aqua", "Emilia", "Kurisu", "Shinobu", "Kanao", "Nezuko",
    "Nobara", "Maki", "Power", "Makima", "Rias", "Akeno", "Lucy", "Erza", "Tohru", "Kobayashi",
    "Kaguya", "Chika", "Ai Hoshino", "Kana Arima", "Frieren", "Fern", "Maomao", "Komi", "Nino Nakano", "Yotsuba Nakano",
    "Holo", "CC", "Eula", "Raiden Shogun", "Ganyu", "Yoimiya", "Hu Tao", "Keqing", "Ayaka", "Nahida",
    "Kafka", "Silver Wolf", "Firefly", "Acheron", "March 7th", "Himeko", "Bronya", "Seele", "Jingliu", "Black Swan",
]

OUTFITS = [
    "maid costume", "school uniform", "office outfit", "hoodie", "casual outfit", "winter coat", "kimono", "sportswear", "idol outfit", "barista uniform",
    "nurse outfit", "chef uniform", "gamer outfit", "detective coat", "fantasy robe", "streetwear", "summer dress", "business suit", "punk outfit", "formal dress",
]

SCENES = [
    "city street", "cafe", "library", "classroom", "office", "park", "rooftop", "train station", "bedroom", "kitchen",
    "rainy alley", "flower garden", "shrine", "snowy street", "beach", "shopping mall", "music stage", "arcade", "studio room", "festival street",
]

LIGHTING = [
    "daylight", "sunset", "night", "soft light", "warm light", "cold light", "indoor light", "window light", "neon light", "overcast",
]


def make_positive(character: str, outfit: str, scene: str, light: str, with_prefix_n: bool) -> str:
    core = f"{character}, full body, {outfit}, 1girl, {scene}, {light}"
    return f"N, {core}" if with_prefix_n else core


def build_prompts():
    prompts = []

    # 50 SFW
    for i in range(50):
        character = CHARACTERS[i % len(CHARACTERS)]
        outfit = OUTFITS[i % len(OUTFITS)]
        scene = SCENES[(i * 3) % len(SCENES)]
        light = LIGHTING[(i * 5) % len(LIGHTING)]
        prompts.append({
            "id": i + 1,
            "type": "SFW",
            "positive": make_positive(character, outfit, scene, light, with_prefix_n=False),
            "negative": NEGATIVE,
        })

    # 50 with N-prefix
    for i in range(50):
        character = CHARACTERS[(i + 17) % len(CHARACTERS)]
        outfit = OUTFITS[(i + 7) % len(OUTFITS)]
        scene = SCENES[(i * 7 + 2) % len(SCENES)]
        light = LIGHTING[(i * 9 + 1) % len(LIGHTING)]
        prompts.append({
            "id": i + 51,
            "type": "N",
            "positive": make_positive(character, outfit, scene, light, with_prefix_n=True),
            "negative": NEGATIVE,
        })

    return prompts


def main():
    data = {
        "version": 2,
        "description": "Calibration prompts for SDXL UNet INT8 (50 SFW + 50 N-prefixed)",
        "prompts": build_prompts(),
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[ok] Saved: {OUT_PATH}")
    print(f"[ok] Total prompts: {len(data['prompts'])}")


if __name__ == "__main__":
    main()
