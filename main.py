import pickle
from helpers import Player, get_settings



with open("dataset/annotations.pkl", "rb") as file:
    group_activity_to_id, player_activity_to_id, annotations = pickle.load(file=file)

settings = get_settings()
import code; code.interact(local=locals())