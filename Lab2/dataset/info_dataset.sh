#!/bin/bash

arquivo="$1"

if [ -z "$arquivo" ]; then
    echo "Uso: $0 <arquivo>"
    exit 1
fi

if [ ! -f "$arquivo" ]; then
    echo "Erro: arquivo '$arquivo' não encontrado."
    exit 1
fi

# Número total de exemplos
exemplos=$(wc -l < "$arquivo")

# Número de features
features=$(head -n 1 "$arquivo" | awk '{print NF-1}')

# Classes existentes
classes=$(cut -d' ' -f1 "$arquivo" | sort -n | uniq)

echo "Arquivo: $arquivo"
echo "Número de exemplos: $exemplos"
echo "Número de features: $features"
echo
echo "Classes e quantidade de exemplos:"

echo "$classes" | while read classe; do
    quantidade=$(grep -c "^$classe " "$arquivo")
    echo "  Classe $classe: $quantidade exemplos"
done

echo
echo "Número de classes: $(echo "$classes" | wc -l)"
