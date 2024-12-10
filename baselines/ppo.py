from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sustaingym.envs.building import BuildingEnv, ParameterGenerator


def main():
    params = ParameterGenerator(building='OfficeSmall', weather='Hot_Dry', location='Tucson')
    # env = BuildingEnv(params)

    vec_env = make_vec_env('sustaingym/Building-v0', env_kwargs={"parameters": params}, n_envs=4, seed=123)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("baselines/ppo")

    del model # remove to demonstrate saving and loading

    model = PPO.load("baselines/ppo")
    env = BuildingEnv(params)
    obs, info = env.reset(seed=123)
    total_rewards = []
    for _ in range(24):
        action, _states = model.predict(obs)
        obs, rewards, _, _, _ = env.step(action)
        total_rewards.append(rewards)

    print(total_rewards)


if __name__ == "__main__":
    main()