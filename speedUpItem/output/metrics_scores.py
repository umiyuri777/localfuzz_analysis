# =============================================================================
# 貼り付け用データ（compare_models.py の出力をここにコピー）
# =============================================================================
METRICS_SCORES: dict[str, dict[str, dict[str, float]]] = {
    "per_run": {
        "BL": {
            "precision": 0.7200,
            "recall": 1.0000,
            "f1": 0.8372
        },
        "LR": {
            "precision": 0.8113,
            "recall": 0.8000,
            "f1": 0.8056
        },
        "DT": {
            "precision": 0.8464,
            "recall": 0.8111,
            "f1": 0.8284
        },
        "RF": {
            "precision": 0.8464,
            "recall": 0.8111,
            "f1": 0.8284
        },
        "GB": {
            "precision": 0.8464,
            "recall": 0.8111,
            "f1": 0.8284
        }
    },
    "bug_detected_any": {
        "BL": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "LR": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "DT": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "RF": {
            "precision": 0.9400,
            "recall": 1.0000,
            "f1": 0.9691
        },
        "GB": {
            "precision": 0.9394,
            "recall": 0.9894,
            "f1": 0.9637
        }
    },
    "bug_detected_all": {
        "BL": {
            "precision": 0.4200,
            "recall": 1.0000,
            "f1": 0.5915
        },
        "LR": {
            "precision": 0.6667,
            "recall": 0.5714,
            "f1": 0.6154
        },
        "DT": {
            "precision": 0.5645,
            "recall": 0.8333,
            "f1": 0.6731
        },
        "RF": {
            "precision": 0.5397,
            "recall": 0.8095,
            "f1": 0.6476
        },
        "GB": {
            "precision": 0.5397,
            "recall": 0.8095,
            "f1": 0.6476
        }
    }
}
