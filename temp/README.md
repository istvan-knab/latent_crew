# 2D MNIST Baseline Classifier (TensorFlow & W&B)

Ez a modul a féléves 3D detekciós és szegmentációs projekt 2D osztályozási baseline alprojektje. A feladat célja egy tiszta, reprodukálható, OOP-szemléletű és moduláris TensorFlow/Keras munkafolyamat kialakítása és naplózása a Weights & Biases felületén.

---

## 1. Könyvtárstruktúra

```text
temp/
├── configs/            # YAML konfigurációk (alapértékek, hiperparaméterek)
├── scripts/            # CLI belépési pontok (train, evaluate)
├── src/                # Üzleti logika (data, model, trainer, evaluator, device)
├── tests/              # PyTest egység- és smoke tesztek
├── outputs/            # Checkpointok (.keras), metrikák (JSON/MD) és ábrák (PNG)
└── requirements.txt    # Függőségek rögzített listája