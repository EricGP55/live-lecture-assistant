#!/bin/bash
# Builds a sample "class" with two voices (teacher and student) using the macOS system voices.
# The spoken content is in Spanish on purpose: it is what the assistant will hear in real classes.
set -e
cd "$(dirname "$0")"
OUT=class_sample.wav
TMP=$(mktemp -d)
TEACHER="Eddy (Español (España))"
STUDENT="Flo (Español (España))"
i=0
line() { i=$((i+1)); say -v "$1" -o "$TMP/$(printf %02d $i).aiff" "$2"; }
line "$TEACHER" "Buenos días. Hoy vamos a empezar con las derivadas. Abrid el cuaderno y apuntad, porque esto es importante y entra en el examen."
line "$TEACHER" "La derivada de una función mide cómo cambia su valor cuando cambia la variable. Formalmente, es el límite del cociente incremental cuando h tiende a cero."
line "$STUDENT" "Profe, una pregunta, ¿esto entra en el parcial del día doce?"
line "$TEACHER" "Buena pregunta. Sí, entra todo lo de hoy. Fijaos bien en la regla de la cadena, porque la voy a preguntar seguro."
line "$TEACHER" "Por ejemplo, la derivada de e elevado a x es e elevado a x. Es la única función que es igual a su propia derivada."
line "$STUDENT" "No lo entiendo, ¿por qué pasa eso con la e?"
line "$TEACHER" "Lo vemos con la definición de límite. Para el jueves, haced los ejercicios del tema tres, del uno al diez."
line "$TEACHER" "Vamos a verlo. Con la definición, el cociente incremental de e elevado a x es e elevado a x por, e elevado a h menos uno, partido por h. Ese límite vale uno cuando h tiende a cero, y por eso la derivada de e elevado a x es e elevado a x."
line "$TEACHER" "¿Alguna pregunta?"
for f in "$TMP"/*.aiff; do ffmpeg -loglevel error -y -i "$f" -ar 16000 -ac 1 "${f%.aiff}.wav"; done
ffmpeg -loglevel error -y -f lavfi -t 0.8 -i anullsrc=r=16000:cl=mono "$TMP/silence.wav"
: > "$TMP/list.txt"
for f in "$TMP"/[0-9]*.wav; do echo "file '$f'" >> "$TMP/list.txt"; echo "file '$TMP/silence.wav'" >> "$TMP/list.txt"; done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$TMP/list.txt" -ar 16000 -ac 1 "$OUT"
rm -rf "$TMP"
echo "OK: tests/$OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT") s)"
