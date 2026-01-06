import random

def get_fire_risk(lat, lon):
    # simulate realistic satellite risk
    weights = ["Low", "Medium", "High"]
    return random.choices(weights, weights=[0.3, 0.3, 0.4])[0]
