"""Experimento: como uma RNN (GRU) se compara à CNN 1D neste problema?

Uso (da raiz do repositório, com os dados já baixados pelo notebook):
    python experimentos/exp_rnn.py                 # as três variantes
    python experimentos/exp_rnn.py cnn_gru         # só uma

Variantes, todas com os mesmos dados, pesos (raiz) e callbacks do notebook:
- gru:          GRU lendo o batimento na ordem normal (QRS primeiro, zeros no fim)
- gru_reverso:  GRU lendo de trás para frente (zeros primeiro, QRS por último)
- cnn_gru:      dois blocos convolucionais resumem o sinal (187 -> 46 passos) e a GRU lê o resumo

A comparação é feita pela macro-F1 na VALIDAÇÃO. Na CPU, cada GRU pura leva ~40 min.
"""
import sys

import keras
from keras import layers

from comum import carregar_dados, pesos_de_classe, treinar_e_avaliar


def cabeca(modelo):
    """Mesma parte final da CNN do notebook."""
    modelo.add(layers.Dropout(0.3))
    modelo.add(layers.Dense(64, activation="relu"))
    modelo.add(layers.Dense(5, activation="softmax"))
    return modelo


def construir_gru(nome, reverso=False):
    modelo = keras.Sequential(
        [keras.Input((187,)), layers.Reshape((187, 1)), layers.GRU(64, go_backwards=reverso)], name=nome)
    return cabeca(modelo)


def construir_cnn_gru(nome):
    modelo = keras.Sequential([keras.Input((187,)), layers.Reshape((187, 1))], name=nome)
    for filtros in [32, 64]:
        modelo.add(layers.Conv1D(filtros, 5, padding="same"))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Activation("relu"))
        modelo.add(layers.MaxPooling1D(2))
    modelo.add(layers.GRU(64))
    return cabeca(modelo)


VARIANTES = {
    "gru": lambda: construir_gru("gru"),
    "gru_reverso": lambda: construir_gru("gru_reverso", reverso=True),
    "cnn_gru": lambda: construir_cnn_gru("cnn_gru"),
}

if __name__ == "__main__":
    escolhidas = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTES)
    dados = carregar_dados()
    for nome in escolhidas:
        treinar_e_avaliar(VARIANTES[nome], dados, pesos_de_classe(dados[2], "raiz"))
