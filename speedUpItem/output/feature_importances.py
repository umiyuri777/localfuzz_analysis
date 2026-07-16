# =============================================================================
# 貼り付け用データ（compare_models.py の出力をここにコピー）
# =============================================================================
FEATURE_IMPORTANCES: dict[str, dict[str, dict[str, float]]] = {
    "per_run": {
        "DT": {
            "cpNum": 0.3351,
            "cpNum_range": 0.6642,
            "cpNum_dir_2": 0.0002,
            "cpNum_dir_3": 0.0000,
            "cpNum_dir_4": 0.0005
        },
        "RF": {
            "cpNum": 0.2805,
            "cpNum_range": 0.7127,
            "cpNum_dir_2": 0.0020,
            "cpNum_dir_3": 0.0020,
            "cpNum_dir_4": 0.0027
        },
        "GB": {
            "cpNum": 0.3342,
            "cpNum_range": 0.6589,
            "cpNum_dir_2": 0.0010,
            "cpNum_dir_3": 0.0030,
            "cpNum_dir_4": 0.0029
        }
    },
    "bug_detected_any": {
        "DT": {
            "cpNum": 0.2634,
            "cpNum_range": 0.7183,
            "cpNum_dir_2": 0.0159,
            "cpNum_dir_3": 0.0017,
            "cpNum_dir_4": 0.0006
        },
        "RF": {
            "cpNum": 0.3049,
            "cpNum_range": 0.6197,
            "cpNum_dir_2": 0.0288,
            "cpNum_dir_3": 0.0236,
            "cpNum_dir_4": 0.0231
        },
        "GB": {
            "cpNum": 0.2837,
            "cpNum_range": 0.6622,
            "cpNum_dir_2": 0.0218,
            "cpNum_dir_3": 0.0061,
            "cpNum_dir_4": 0.0261
        }
    },
    "bug_detected_all": {
        "DT": {
            "cpNum": 0.3702,
            "cpNum_range": 0.6184,
            "cpNum_dir_2": 0.0018,
            "cpNum_dir_3": 0.0013,
            "cpNum_dir_4": 0.0082
        },
        "RF": {
            "cpNum": 0.3367,
            "cpNum_range": 0.6384,
            "cpNum_dir_2": 0.0059,
            "cpNum_dir_3": 0.0066,
            "cpNum_dir_4": 0.0125
        },
        "GB": {
            "cpNum": 0.3716,
            "cpNum_range": 0.6141,
            "cpNum_dir_2": 0.0022,
            "cpNum_dir_3": 0.0023,
            "cpNum_dir_4": 0.0098
        }
    }
}
