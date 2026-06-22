# =============================================================================
# 貼り付け用データ（compare_models.py の出力をここにコピー）
# =============================================================================
FEATURE_IMPORTANCES: dict[str, dict[str, dict[str, float]]] = {
    "per_run": {
        "DT": {
            "cpNum": 0.3351,
            "cpNum_range": 0.6642
        },
        "RF": {
            "cpNum": 0.2805,
            "cpNum_range": 0.7127
        },
        "GB": {
            "cpNum": 0.3342,
            "cpNum_range": 0.6589
        }
    },
    "bug_detected_any": {
        "DT": {
            "cpNum": 0.2634,
            "cpNum_range": 0.7183
        },
        "RF": {
            "cpNum": 0.3049,
            "cpNum_range": 0.6197
        },
        "GB": {
            "cpNum": 0.2837,
            "cpNum_range": 0.6622
        }
    },
    "bug_detected_all": {
        "DT": {
            "cpNum": 0.3702,
            "cpNum_range": 0.6184
        },
        "RF": {
            "cpNum": 0.3367,
            "cpNum_range": 0.6384
        },
        "GB": {
            "cpNum": 0.3716,
            "cpNum_range": 0.6141
        }
    }
}
