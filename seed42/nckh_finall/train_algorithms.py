"""
Thuật toán ML/DL — tất cả import ở đầu file theo yêu cầu.
- ML: Decision Tree, Random Forest, Logistic Regression, MLP(sklearn)
- DL: CNN (Keras) — nếu không có TensorFlow, trả về []
"""

from __future__ import annotations

# ====== ML (sklearn) ======
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ====== DL (Keras) — import an toàn ======
try:
    import tensorflow as tf
    layers = tf.keras.layers
    models = tf.keras.models
    optimizers = tf.keras.optimizers
    losses = tf.keras.losses
    metrics = tf.keras.metrics

    _HAS_TF = True
except Exception:
    _HAS_TF = False


def get_ml_algorithms():
    algos = []

    def DecisionTree():
        return DecisionTreeClassifier( max_depth=12, min_samples_leaf=2,random_state=42), "Decision Tree"

    def RandomForest():
        return RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=2,oob_score=True, random_state=42, n_jobs=-1), "Random Forest"

    def LogisticReg():
        # Tránh n_jobs với solver mặc định để không lỗi
        return LogisticRegression(penalty="l2",C=0.5, solver="lbfgs",max_iter=1000), "Logistic Regression"   # C nhỏ hơn → phạt mạnh hơn → bớt overfit
  
    def MLP_sklearn():
        # Cấu hình nhẹ để đảm bảo thời gian train hợp lý
        return MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                             batch_size=256, max_iter=200, random_state=42,
                             early_stopping=True), "MLP(sklearn)"

    algos += [DecisionTree, RandomForest, LogisticReg, MLP_sklearn]
    return algos


def get_dl_algorithms():
    if not _HAS_TF:
        return []

    def CNN(input_shape, n_classes):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),                    # ⟵ THAY cho GlobalAveragePooling2D
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy(name="acc")],
        )
        return model, "CNN"
    
    return [CNN]

