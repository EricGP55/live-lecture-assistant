"""Transcripción con Parakeet v3 (MLX), en modo frase completa.

El modo streaming de parakeet-mlx resultó lento (RTF > 1) e impreciso en un M1 8 GB,
así que se transcribe cada frase cortada por el VAD (RTF ~0,05) y se dan parciales
re-transcribiendo la frase en curso cada pocos segundos.
"""
import warnings

import mlx.core as mx
import numpy as np

warnings.filterwarnings("ignore")


class Transcriptor:
    def __init__(self, modelo: str):
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel
        self._mel = get_logmel
        self.m = from_pretrained(modelo)
        self.transcribir(np.zeros(16000, dtype=np.float32))  # calentar Metal

    def transcribir(self, audio: np.ndarray) -> str:
        if len(audio) < 1600:  # < 0,1 s
            return ""
        # Sin un poco de silencio delante, Parakeet tiende a perder la primera palabra.
        pad = np.zeros(int(0.3 * 16000), dtype=np.float32)
        audio = np.concatenate([pad, audio.astype(np.float32), pad[:3200]])
        mel = self._mel(mx.array(audio), self.m.preprocessor_config)
        r = self.m.generate(mel)[0]
        mx.clear_cache()
        return r.text.strip()
