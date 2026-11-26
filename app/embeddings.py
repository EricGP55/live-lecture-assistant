"""Local multilingual text embeddings: multilingual-e5-small (quantized ONNX), no PyTorch.

e5 expects prefixes: "passage: " for stored text and "query: " for questions.
"""
import threading

import numpy as np

from .config import ROOT

MODEL_DIR = ROOT / "models" / "e5-small"
DIM = 384


class Embedder:
    def __init__(self, model_dir=MODEL_DIR, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.lock = threading.Lock()

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        with self.lock:
            hidden = self.sess.run(None, feed)[0]           # (batch, tokens, 384)
        m = mask[..., None].astype(np.float32)
        v = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)  # mean pooling
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"passage: {t}" for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._embed([f"query: {text}"])[0]
