import json


with open("/Data/meierj/YoutubeVIS2021/annotations/youtube_vis_2021_train.json", "r") as file:
    ann = json.load(file)

# print(ann["categories"][0:10])

# print("\n\n\n")

# print(ann["videos"][0:10])

# print("\n\n\n")

# print(ann["images"][0:10])

# print("\n\n\n")

print(ann["annotations"][-9:-1])