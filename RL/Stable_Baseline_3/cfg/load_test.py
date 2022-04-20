import yaml

with open("contGrid_v2_PPO.yaml", "r") as file:
    try:
        cfg = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# # print(stream["params"])
# stream = file("contGrid_v2_PPO.yaml", 'r')
# cfg = yaml.load(stream)
grid_dims = cfg["params"]["grid_dims"]
print(cfg["params"]["grid_dims"])
print(grid_dims)
