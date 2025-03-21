import soundfile as sf
import numpy as np
from kokoro import KPipeline
from googletrans import Translator


async def translate_text(text: str, src: str = 'auto', dest: str = 'hi') -> str:
    async with Translator() as translator:
        result = await translator.translate(text, src=src, dest=dest)
        return result.text


def hindi_tts(text, output_path):
    pipeline = KPipeline(lang_code='h')  # <= make sure lang_code matches voice

    generator = pipeline(
        text, voice='hf_alpha',  # <= change voice here
        speed=1, split_pattern=r'\n+'
    )

    audio_segments = []

    for i, (_, _, audio) in enumerate(generator):
        arr = audio.cpu().numpy() if hasattr(audio, "cpu") else audio.numpy()
        audio_segments.append(arr)

    merged_audio = np.concatenate(audio_segments, axis=0)
    sf.write(output_path, merged_audio, 24000)
