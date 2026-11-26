# Asistente de clase en vivo

Graba la clase, la transcribe en directo y distingue **profesor (P) / alumno (A)**. Todo en local, en el Mac.
Diseño completo y decisiones: [DISENO.md](DISENO.md).

**Estado:** fase 1 hecha (transcripción en vivo + voces + historial + audio). Siguiente: fase 2 (preguntas a la IA).

## Instalación (una vez)

```bash
cd clase-ia
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models
curl -L -o models/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -L -o models/speaker.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```

La primera vez se descarga el modelo Parakeet v3 (~1,2 GB) en `~/.cache/huggingface`.

## Uso

```bash
.venv/bin/python clase.py --asignatura "Cálculo"
```

- Abre `http://127.0.0.1:8765/` y la ventana mini en `/mini`.
- **Ctrl+C** o el botón "Terminar clase" cierran la clase. Al cerrarse, se guardan el texto, el audio y la huella del profesor.
- La primera vez, macOS pedirá **permiso de micrófono** para la Terminal.

Otras opciones:

```bash
.venv/bin/python clase.py --dispositivos                               # lista micros
.venv/bin/python clase.py --asignatura "Cálculo" --dispositivo "USB"   # micro externo
.venv/bin/python clase.py --asignatura Demo --archivo tests/clase_prueba.wav   # simula una clase
```

## Datos

| Ruta | Qué hay |
|---|---|
| `datos/clase.db` | SQLite: asignaturas, clases, frases (con búsqueda FTS5) y huella del profesor por asignatura |
| `datos/audio/*.opus` | Audio de cada clase (~11 MB por hora) |

## Pruebas

```bash
tests/make_test_audio.sh                     # genera una clase de prueba con dos voces del sistema
.venv/bin/python tests/prueba_pipeline.py    # la procesa sin web y muestra el resultado
```

## Rendimiento medido (MacBook Air M1, 8 GB)

- **Transcripción:** unas 20× más rápida que el tiempo real.
- **Frase final:** aparece ~1–1,5 s después de que se termine de decir (0,5 s de pausa + 0,3–0,9 s de proceso).
- **Texto provisional:** se actualiza cada 1,5 s mientras alguien habla.
- **RAM:** ~1,4 GB en marcha (pico de 2,5 GB al cargar el modelo).
