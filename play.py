from sustaingym.envs.building import BuildingEnv, ParameterGenerator

params = ParameterGenerator(
    building='OfficeSmall', weather='Hot_Dry', location='Tucson')
env = BuildingEnv(params)

# Print observation space
print(env.observation_space.shape)
# Print action space
print(env.action_space.shape)
action = env.action_space.sample()
for i in range(10):
    num_hours = 24
    obs, _ = env.reset(seed=123)
    total_reward = 0
    for _ in range(num_hours):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(total_reward)