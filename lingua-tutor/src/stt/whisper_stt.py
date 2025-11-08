# src/stt/whisper_stt.py
import whisper
import torch
from pathlib import Path
from loguru import logger


from datetime import datetime

from src.utils import timer_decorator


@timer_decorator
def transcribe_and_save(
    audio_path: Path,
    output_dir: Path = Path("output/stt"),
    model_name: str = "base",
    language: str | None = "en",
) -> Path:
    """
    Transcribes an audio file and saves the result to a file with a timestamp.
    The output filename is generated automatically based on the input file and a timestamp.

    Args:
        audio_path: Path to the audio file to transcribe.
        output_dir: Directory to save the transcription file.
        model_name: The name of the Whisper model to use.
        language: The language of the audio. Set to None for auto-detection.

    Returns:
        The absolute path of the saved file.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # --- Filename and Path Generation ---
    input_stem = audio_path.stem
    # Timestamp format: YYYYMMDD_HHMMSS_microseconds(first 2 digits)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-4]
    output_filename = f"{input_stem}_{timestamp}.txt"
    output_path = output_dir / output_filename

    # --- Transcription Logic ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name, device=device)

    lang_info = f"Language: {language}" if language else "Language: auto-detect"
    logger.info(f"Transcribing audio file: {audio_path} ({lang_info})...")
    result = model.transcribe(str(audio_path), language=language)
    transcribed_text = result["text"].strip()

    logger.info("----- Transcription Result -----")
    logger.info(transcribed_text)
    logger.info("--------------------------------")

    # --- Save Logic ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcribed_text, encoding="utf-8")

    final_path = output_path.resolve()
    logger.success(f"Transcription saved to: {final_path}")

    return final_path


def _run():
    audio_file = Path("test_data/this_is_a_simple_test.wav")

    final_path = transcribe_and_save(
        audio_path=audio_file,
    )
    logger.info(f"\n{final_path}")


if __name__ == "__main__":
    _run()
