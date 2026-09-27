"""Experimento: qual intensidade de peso de classe usar na CNN 1D?

Uso (da raiz do repositório, com os dados já baixados pelo notebook):
    python experimentos/exp_pesos.py

Treina a mesma CNN do notebook com pesos "balanceado", "raiz" e "nenhum".
A escolha é feita pela macro-F1 na VALIDAÇÃO. Leva ~30 min na CPU.
"""
import keras
from keras import layers

from comum import carregar_dados, pesos_de_classe, treinar_e_avaliar


def construir_cnn(nome):
    modelo = keras.Sequential([keras.Input((187,)), layers.Reshape((187, 1))], name=nome)
    for filtros in [32, 64, 128, 128]:
        modelo.add(layers.Conv1D(filtros, 5, padding="same"))
        modelo.add(layers.BatchNormalization())
        modelo.add(layers.Activation("relu"))
        modelo.add(layers.MaxPooling1D(2))
    modelo.add(layers.GlobalAveragePooling1D())
    modelo.add(layers.Dropout(0.3))
    modelo.add(layers.Dense(64, activation="relu"))
    modelo.add(layers.Dense(5, activation="softmax"))
    return modelo


if __name__ == "__main__":
    dados = carregar_dados()
    for modo in ["balanceado", "raiz", "nenhum"]:
        treinar_e_avaliar(lambda: construir_cnn(f"cnn_pesos_{modo}"), dados, pesos_de_classe(dados[2], modo))
