# Live Class Assistant

Records a class, transcribes it live, tells apart the **teacher (T)** from the **students (S)**, and lets you **ask questions about what was said** and **keep track of your doubts** while the class is going on. Audio and transcription stay on the Mac; only short transcript excerpts are sent to Claude when you ask something.

**Status:** phases 1–3 done (live transcription, speakers, history, audio, questions to the AI, doubts mode). Next: phase 4 (alerts with your own rules).

## Setup (once)

```bash
cd clase-ia
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models
curl -L -o models/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -L -o models/speaker.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
mkdir -p models/e5-small
curl -L -o models/e5-small/model.onnx https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/onnx/model_quantized.onnx
curl -L -o models/e5-small/tokenizer.json https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/tokenizer.json
```

Questions use your **Claude subscription**: you only need to be logged in to Claude Code (`claude` in a terminal, then `/login`). No API key is needed.

On first run, the Parakeet v3 model (~1.2 GB) is downloaded to `~/.cache/huggingface`.

## Usage

```bash
.venv/bin/python main.py --subject "Calculus"
```

- Opens `http://127.0.0.1:8765/` (full page) and `/mini` (small window).
- **Ctrl+C** or the "End class" button closes the class. The text, the audio and the teacher's voiceprint are saved.
- On first run, macOS asks for **microphone permission** for the Terminal.

Other options:

```bash
.venv/bin/python main.py --devices                                  # list microphones
.venv/bin/python main.py --subject "Calculus" --device "USB"        # external microphone
.venv/bin/python main.py --subject Demo --file tests/class_sample.wav   # simulate a class
```

Settings live in [config.yaml](config.yaml): microphone, VAD pauses, speaker thresholds, AI model, port and data folder.

### Asking questions

- **Where:** in the right-hand panel of the main page, or in the mini window.
- **Scope:** *this class*, *all classes of this subject* or *all my classes*.
- **Answers:** they stream in, in the language of the question, and cite the moment they come from as `[mm:ss]`. Click a citation to jump to that sentence in the transcript.
- If the class does not cover the question, the answer says so, and any general explanation is labelled as coming from outside the class.

### Doubts mode

Open the **Doubts** tab, or switch the mini window to *Doubt*, and write what you did not understand.

- **Check:** the app tells you at once whether the class has already answered it (🟢 answered / 🟡 partly / 🔴 not yet), with what was said, what is missing and 1–2 questions you could ask. Each question has a *Copy* button.
- **Watch:** while the teacher keeps talking, every new sentence is compared locally with your open doubts. Only a clear match is confirmed with Claude, and then you get a silent macOS notification and an alert on the page: "your doubt is being answered (mm:ss)".
- **Good moment to ask:** when the teacher asks "¿alguna pregunta?" / "any questions?", or pauses after speaking, and you still have open doubts, you get a heads-up.
- **Suggest questions:** gives 2–3 questions for right now, based on the last minutes, your open doubts and your **Profile**.
- **Profile tab:** what worries you in this subject, plus whether the questions should use *tú* or *usted* and be short or linked to what was said. It is saved per subject.
- **Carried over:** unresolved doubts appear again in the next class of the same subject.

## How it works

1. **Audio:** microphone or file at 16 kHz mono, also saved as Opus (~11 MB per hour).
2. **VAD:** silero (sherpa-onnx) cuts the audio into sentences at each pause.
3. **Transcription:** Parakeet v3 (MLX) transcribes each full sentence, about 20× faster than real time. While someone is speaking, the sentence in progress is re-transcribed every 1.5 s as provisional text.
4. **Speakers:** a voice embedding (CAM++, sherpa-onnx) groups sentences by voice. That is combined with text cues ("abrid el cuaderno" vs. "profe, una pregunta") and turn-taking (a teacher's "good question" marks the previous sentence as a student's) into a probability P(teacher).
   - The teacher's voiceprint is saved per subject, so the next class recognises them from the first sentence.
5. **Storage:** SQLite with full-text search (FTS5), plus a local web page updated over a WebSocket.
6. **Questions (phase 2):**
   - **Embeddings:** each sentence gets a multilingual embedding (multilingual-e5-small, quantized ONNX, ~4 ms per sentence, no PyTorch).
   - **Search:** a question is matched with hybrid search (embeddings + BM25, fused with Reciprocal Rank Fusion). The best sentences are expanded with their neighbours and with the last few sentences of the class.
   - **Prompt size:** everything sent is packed into ~1,300 tokens.
   - **Claude:** answers come through the Claude Agent SDK using your subscription. Each question runs in a fresh session, with no tools and without your account's connectors, which keeps the overhead at ~600 tokens. The next session is opened in advance, so the first words arrive in ~0.6–1.4 s.
   - **One shared session:** a waiting session uses ~150 MB of RAM, so questions, doubts and suggestions all share it; each task puts its own instructions at the top of the message.
7. **Doubts (phase 3):**
   - **Watching:** each new teacher sentence is compared with every open doubt using the same local embeddings. On the sample class, sentences that really address a doubt score ≥ 0.89 and the rest ≤ 0.85.
   - **Confirming:** only strong or unusual matches are confirmed with a short Claude call, so a doubt the class never covers costs nothing after the first check.
   - **Thresholds:** they are in `config.yaml` under `doubts:`.

## Data

| Path | Contents |
|---|---|
| `data/class.db` | SQLite: subjects, classes, sentences (with FTS5 search) and the teacher voiceprint per subject |
| `data/audio/*.opus` | Audio of each class |

Questions and answers are stored in the `questions` table of `data/class.db`, together with the sentences used as context and the tokens spent. Doubts, with their status, evidence, suggested questions and tokens, are in the `doubts` table.

## Private files and the commit guard

`DISENO.md` (the private design notes), `data/` and `models/` are git-ignored. On top of that,
a pre-commit hook in `.githooks/` refuses any commit that adds them, or that contains something
that looks like a key or token. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

The list of blocked paths is in `.githooks/blocked-paths`.

## Tests

```bash
tests/make_test_audio.sh                    # builds a sample class with two system voices
.venv/bin/python tests/test_pipeline.py     # runs it through the pipeline, without the web page
.venv/bin/python tests/test_qa.py           # transcribes it and asks 4 questions through your subscription
.venv/bin/python tests/test_doubts.py       # replays it live with 3 doubts: check, watch, suggest
```

## Measured performance (MacBook Air M1, 8 GB)

- **Transcription:** about 20× faster than real time.
- **Final sentence:** appears ~1–1.5 s after it is spoken (0.5 s pause + 0.3–0.9 s processing).
- **Provisional text:** refreshed every 1.5 s while someone is speaking.
- **RAM:** ~1.4 GB while running (2.5 GB peak while loading the model).
- **Question → first words:** ~0.5–1.4 s. Each question costs ~1,300–1,500 input + 50–200 output tokens.
- **Doubts:** the first check costs ~1,400–2,300 tokens. Each confirmation while watching costs ~500–1,000, and only happens on a clear match.
