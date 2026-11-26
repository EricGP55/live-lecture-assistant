"""Speech-to-text with Parakeet v3 (MLX), one full sentence at a time.

parakeet-mlx's streaming mode was slow (RTF > 1) and inaccurate on an 8 GB M1, so each
sentence cut by the VAD is transcribed as a whole (RTF ~0.05), and partial results come
from re-transcribing the sentence in progress every few seconds.
"""
import warnings

import mlx.core as mx
import numpy as np

warnings.filterwarnings("ignore")


class Transcriber:
    def __init__(self, model: str):
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel
        self._logmel = get_logmel
        self.model = from_pretrained(model)
        self.transcribe(np.zeros(16000, dtype=np.float32))  # warm up Metal

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < 1600:  # < 0.1 s
            return ""
        # Without a little leading silence, Parakeet tends to drop the first word.
        pad = np.zeros(int(0.3 * 16000), dtype=np.float32)
        audio = np.concatenate([pad, audio.astype(np.float32), pad[:3200]])
        mel = self._logmel(mx.array(audio), self.model.preprocessor_config)
        result = self.model.generate(mel)[0]
        mx.clear_cache()
        return result.text.strip()
