def classify_anemia(hb: float):

    if hb >= 13:
        return "Normal"

    elif hb >= 11:
        return "Mild Anemia"

    elif hb >= 8:
        return "Moderate Anemia"

    else:
        return "Severe Anemia"