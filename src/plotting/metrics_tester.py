import pickle

with open("metrics/metrics.pkl", "rb") as f:
    data = pickle.load(f)

for key, val in data.items():
    print(f"{key}: {len(val)}")
    if key == "episode_lengths":
        print(val)
