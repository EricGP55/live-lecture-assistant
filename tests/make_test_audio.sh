#!/bin/bash
# Genera una "clase" de prueba con dos voces (profesor y alumna) usando la voz del sistema.
set -e
cd "$(dirname "$0")"
OUT=clase_prueba.wav
TMP=$(mktemp -d)
P="Eddy (Español (España))"
A="Flo (Español (España))"
i=0
line() { i=$((i+1)); say -v "$1" -o "$TMP/$(printf %02d $i).aiff" "$2"; }
line "$P" "Buenos días. Hoy vamos a empezar con las derivadas. Abrid el cuaderno y apuntad, porque esto es importante y entra en el examen."
line "$P" "La derivada de una función mide cómo cambia su valor cuando cambia la variable. Formalmente, es el límite del cociente incremental cuando h tiende a cero."
line "$A" "Profe, una pregunta, ¿esto entra en el parcial del día doce?"
line "$P" "Buena pregunta. Sí, entra todo lo de hoy. Fijaos bien en la regla de la cadena, porque la voy a preguntar seguro."
line "$P" "Por ejemplo, la derivada de e elevado a x es e elevado a x. Es la única función que es igual a su propia derivada."
line "$A" "No lo entiendo, ¿por qué pasa eso con la e?"
line "$P" "Lo vemos con la definición de límite. Para el jueves, haced los ejercicios del tema tres, del uno al diez."
for f in "$TMP"/*.aiff; do ffmpeg -loglevel error -y -i "$f" -ar 16000 -ac 1 "${f%.aiff}.wav"; done
ffmpeg -loglevel error -y -f lavfi -t 0.8 -i anullsrc=r=16000:cl=mono "$TMP/sil.wav"
: > "$TMP/list.txt"
for f in "$TMP"/[0-9]*.wav; do echo "file '$f'" >> "$TMP/list.txt"; echo "file '$TMP/sil.wav'" >> "$TMP/list.txt"; done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$TMP/list.txt" -ar 16000 -ac 1 "$OUT"
rm -rf "$TMP"
echo "OK: tests/$OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT") s)"
