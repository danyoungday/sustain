from stable_baselines3 import PPO
from sustaingym.envs.building import BuildingEnv, ParameterGenerator


def main():
    params = ParameterGenerator(building='OfficeSmall', weather='Hot_Dry', location='Tucson')
    env = BuildingEnv(params)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("baselines/ppo")

    del model # remove to demonstrate saving and loading

    model = PPO.load("baselines/ppo")
    env = BuildingEnv(params)
    avg_reward = 0
    for i in range(10):
        obs, info = env.reset(seed=i)
        for _ in range(24):
            action, _states = model.predict(obs)
            obs, reward, _, _, _ = env.step(action)
            avg_reward += reward

    print(avg_reward / 10)


if __name__ == "__main__":
    main()