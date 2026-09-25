"""Código compartilhado pelos experimentos: mesmos dados, divisão, callbacks e métricas do notebook."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heartbeat"
CLASSES = ["N", "S", "V", "F", "Q"]


def carregar_dados():
    """Mesma divisão do notebook: 15% do treino para validação, estratificado, semente 42."""
    treino = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None).to_numpy("float32")
    teste = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None).to_numpy("float32")
    X, y = treino[:, :-1], treino[:, -1].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    return X_tr, X_val, y_tr, y_val, teste[:, :-1], teste[:, -1].astype(int)


def pesos_de_classe(y_tr, modo):
    balanceados = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=y_tr)
    pesos = {"balanceado": balanceados, "raiz": np.sqrt(balanceados), "nenhum": np.ones(len(CLASSES))}[modo]
    return dict(enumerate(pesos))


class Relogio(keras.callbacks.Callback):
    """Imprime a macro-F1 de validação e o tempo acumulado a cada época."""

    def on_train_begin(self, logs=None):
        self.inicio = time.time()

    def on_epoch_end(self, epoca, logs=None):
        print(f"  {self.model.name} época {epoca + 1}: val_f1={logs['val_f1_macro']:.3f} "
              f"({time.time() - self.inicio:.0f} s)", flush=True)


def treinar_e_avaliar(construtor, dados, class_weight, epocas=40):
    """Treina como no notebook e devolve as métricas. A decisão usa a VALIDAÇÃO; o teste é só referência."""
    X_tr, X_val, y_tr, y_val, X_te, y_te = dados
    keras.utils.set_random_seed(42)  # antes de construir, para fixar a inicialização dos pesos
    modelo = construtor()
    modelo.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.F1Score(average="macro", name="f1_macro")],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1_macro", mode="max", factor=0.5, patience=2),
        Relogio(),
    ]
    inicio = time.time()
    h = modelo.fit(
        X_tr, keras.utils.to_categorical(y_tr, len(CLASSES)),
        validation_data=(X_val, keras.utils.to_categorical(y_val, len(CLASSES))),
        epochs=epocas, batch_size=256, class_weight=class_weight, callbacks=callbacks, verbose=0,
    )
    segundos = time.time() - inicio
    pred_val = modelo.predict(X_val, batch_size=1024, verbose=0).argmax(1)
    pred_te = modelo.predict(X_te, batch_size=1024, verbose=0).argmax(1)
    resultado = {
        "modelo": modelo.name,
        "parametros": modelo.count_params(),
        "epocas": len(h.history["loss"]),
        "seg_por_epoca": round(segundos / len(h.history["loss"])),
        "val_macroF1": round(f1_score(y_val, pred_val, average="macro"), 4),
        "teste_macroF1": round(f1_score(y_te, pred_te, average="macro"), 4),
        "teste_acuracia": round(accuracy_score(y_te, pred_te), 4),
        "teste_revocacao": dict(zip(CLASSES, recall_score(y_te, pred_te, average=None).round(3).tolist())),
    }
    print("RESULTADO " + json.dumps(resultado, ensure_ascii=False), flush=True)
    return resultado
