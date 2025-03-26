"""Text-to-speech and translation module for NewsByte application.

This module handles text translation and Hindi text-to-speech conversion
for the final sentiment analysis output.
"""

import soundfile as sf
import numpy as np
from kokoro import KPipeline
from googletrans import Translator
from config import logger


async def translate_text(text: str, src: str = 'auto', dest: str = 'hi') -> str:
    """Translate text to Hindi using Google Translate.

    Args:
        text (str): Text to translate
        src (str, optional): Source language code. Defaults to 'auto'.
        dest (str, optional): Destination language code. Defaults to 'hi'.

    Returns:
        str: Translated text in Hindi
    """
    async with Translator() as translator:
        logger.debug("Translating text to Hindi.")
        result = await translator.translate(text, src=src, dest=dest)
        return result.text


def hindi_tts(text, output_path):
    """Convert Hindi text to speech using Kokoro TTS.

    Args:
        text (str): Hindi text to convert to speech
        output_path (str): Path where the audio file should be saved

    Raises:
        ValueError: If no audio segments are generated
    """
    pipeline = KPipeline(lang_code='h', repo_id='hexgrad/Kokoro-82M')
    logger.debug("Running Hindi TTS.")
    generator = pipeline(
        text, voice='hf_alpha',
        speed=1, split_pattern=r'\n+'
    )

    audio_segments = []
    for i, (_, _, audio) in enumerate(generator):
        arr = audio.cpu().numpy() if hasattr(audio, "cpu") else audio.numpy()
        audio_segments.append(arr)

    if not audio_segments:
        raise ValueError(
            "No audio segments generated; cannot concatenate empty list.")

    merged_audio = np.concatenate(audio_segments, axis=0)
    sf.write(output_path, merged_audio, 24000)
