from sustaingym.envs.building import BuildingEnv, ParameterGenerator

params = ParameterGenerator(
    building='OfficeSmall', weather='Hot_Dry', location='Tucson')
env = BuildingEnv(params)

# Print observation space
print(env.observation_space.shape)
# Print action space
print(env.action_space.shape)

num_hours = 24
obs, _ = env.reset(seed=123)
rewards = []
for _ in range(num_hours):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
