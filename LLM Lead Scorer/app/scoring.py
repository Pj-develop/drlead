def calculate_lead_score(data):
    score = 0
    if data.get("Industry") == "Tech":
        score += 10
    if data.get("Company Size") > 500:
        score += 5
    if data.get("Monthly Budget for Accommodation") > 50:
        score += 8
    return score
