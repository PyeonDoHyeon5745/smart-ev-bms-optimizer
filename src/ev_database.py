"""
현대·기아 공식 EV 배터리 셀 정보 데이터베이스
출처: 현대자동차/기아 전기차 배터리 셀 정보 구매자 제공 안내사항 (공식 PDF)
"""

EV_DATABASE = {
    "현대": {
        "아이오닉 5": {
            "롱레인지 2WD":  {"kwh": 84.0, "voltage": 697, "motor_kw": 168,  "maker": "SK ON",      "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 AWD":  {"kwh": 84.0, "voltage": 697, "motor_kw": 239,  "maker": "SK ON",      "chemistry": "NCM", "cell": "파우치형"},
            "스탠다드 2WD":  {"kwh": 63.0, "voltage": 523, "motor_kw": 124.9,"maker": "SK ON",      "chemistry": "NCM", "cell": "파우치형"},
            "N":            {"kwh": 84.0, "voltage": 697, "motor_kw": 448,  "maker": "SK ON",      "chemistry": "NCM", "cell": "파우치형"},
        },
        "아이오닉 6": {
            "롱레인지 2WD":  {"kwh": 77.4, "voltage": 697, "motor_kw": 168,  "maker": "LG 에너지솔루션", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 AWD":  {"kwh": 77.4, "voltage": 697, "motor_kw": 239,  "maker": "LG 에너지솔루션", "chemistry": "NCM", "cell": "파우치형"},
            "스탠다드 2WD":  {"kwh": 53.0, "voltage": 480, "motor_kw": 111,  "maker": "LG 에너지솔루션", "chemistry": "NCM", "cell": "파우치형"},
            "N":            {"kwh": 84.0, "voltage": 697, "motor_kw": 448,  "maker": "SK ON",          "chemistry": "NCM", "cell": "파우치형"},
        },
        "더 뉴 아이오닉 6": {
            "롱레인지 2WD":  {"kwh": 84.0, "voltage": 697, "motor_kw": 168,  "maker": "LG 에너지솔루션", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 AWD":  {"kwh": 84.0, "voltage": 697, "motor_kw": 239,  "maker": "LG 에너지솔루션", "chemistry": "NCM", "cell": "파우치형"},
            "스탠다드 2WD":  {"kwh": 63.0, "voltage": 523, "motor_kw": 125,  "maker": "LG 에너지솔루션", "chemistry": "NCM", "cell": "파우치형"},
        },
        "아이오닉 9": {
            "항속형 2WD":   {"kwh": 110.3, "voltage": 610, "motor_kw": 160, "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "항속형 AWD":   {"kwh": 110.3, "voltage": 610, "motor_kw": 226, "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "성능형 AWD":   {"kwh": 110.3, "voltage": 610, "motor_kw": 315, "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
        },
        "코나 일렉트릭": {
            "롱레인지":     {"kwh": 64.8, "voltage": 358, "motor_kw": 150, "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
            "스탠다드":     {"kwh": 48.6, "voltage": 269, "motor_kw": 99,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
        },
        "캐스퍼 일렉트릭": {
            "기본형":       {"kwh": 42.0, "voltage": 266, "motor_kw": 71.1, "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "항속형":       {"kwh": 49.0, "voltage": 310, "motor_kw": 84.5, "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "크로스":       {"kwh": 49.0, "voltage": 310, "motor_kw": 84.5, "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
        },
        "ST1": {
            "기본형":       {"kwh": 76.1,  "voltage": 632, "motor_kw": 160, "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
        },
        "포터Ⅱ 일렉트릭": {
            "기본형":       {"kwh": 60.4,  "voltage": 334, "motor_kw": 135, "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
        },
    },
    "기아": {
        "레이 EV": {
            "기본형":       {"kwh": 35.2, "voltage": 265, "motor_kw": 64.3, "maker": "CATL",  "chemistry": "LFP", "cell": "각형"},
        },
        "니로 EV": {
            "기본형":       {"kwh": 64.8, "voltage": 358, "motor_kw": 150,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
        },
        "EV3": {
            "스탠다드":     {"kwh": 58.3, "voltage": 369, "motor_kw": 150,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 2WD": {"kwh": 81.4, "voltage": 343, "motor_kw": 150,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 4WD": {"kwh": 81.4, "voltage": 343, "motor_kw": 195,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "GT":          {"kwh": 81.4, "voltage": 343, "motor_kw": 215,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
        },
        "EV4": {
            "스탠다드":     {"kwh": 58.3, "voltage": 369, "motor_kw": 150,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지":     {"kwh": 81.4, "voltage": 343, "motor_kw": 150,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
            "GT":          {"kwh": 81.4, "voltage": 343, "motor_kw": 215,  "maker": "HLIGP", "chemistry": "NCM", "cell": "파우치형"},
        },
        "EV5": {
            "스탠다드":     {"kwh": 60.3, "voltage": 299, "motor_kw": 115,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
            "롱레인지 2WD": {"kwh": 81.4, "voltage": 403, "motor_kw": 160,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
            "롱레인지 4WD": {"kwh": 81.4, "voltage": 403, "motor_kw": 195,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
            "GT":          {"kwh": 81.4, "voltage": 403, "motor_kw": 225,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
        },
        "EV6": {
            "스탠다드":     {"kwh": 63.0, "voltage": 523, "motor_kw": 125,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 2WD": {"kwh": 84.0, "voltage": 697, "motor_kw": 169,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 4WD": {"kwh": 84.0, "voltage": 697, "motor_kw": 239,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "GT":          {"kwh": 84.0, "voltage": 697, "motor_kw": 448,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
        },
        "EV9": {
            "스탠다드":     {"kwh": 76.1, "voltage": 632, "motor_kw": 160,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 2WD": {"kwh": 99.8, "voltage": 552, "motor_kw": 150,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "롱레인지 4WD": {"kwh": 99.8, "voltage": 552, "motor_kw": 283,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
            "GT":          {"kwh": 99.8, "voltage": 552, "motor_kw": 374,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
        },
        "PV5": {
            "스탠다드":     {"kwh": 51.5, "voltage": 290, "motor_kw": 89.4, "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
            "롱레인지":     {"kwh": 71.2, "voltage": 402, "motor_kw": 120,  "maker": "CATL",  "chemistry": "NCM", "cell": "각형"},
        },
        "봉고 EV": {
            "기본형":       {"kwh": 60.4, "voltage": 334, "motor_kw": 135,  "maker": "SK ON", "chemistry": "NCM", "cell": "파우치형"},
        },
    },
}


def get_brands():
    return list(EV_DATABASE.keys())


def get_models(brand: str):
    return list(EV_DATABASE.get(brand, {}).keys())


def get_trims(brand: str, model: str):
    return list(EV_DATABASE.get(brand, {}).get(model, {}).keys())


def get_spec(brand: str, model: str, trim: str) -> dict:
    return EV_DATABASE.get(brand, {}).get(model, {}).get(trim, {})


def get_all_models_flat():
    """브랜드-모델-트림 전체 평탄화 리스트"""
    rows = []
    for brand, models in EV_DATABASE.items():
        for model, trims in models.items():
            for trim, spec in trims.items():
                rows.append({
                    "brand": brand,
                    "model": model,
                    "trim": trim,
                    "display": f"{brand} {model} {trim}",
                    **spec,
                })
    return rows
