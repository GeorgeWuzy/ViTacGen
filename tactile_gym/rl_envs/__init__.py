from gym.envs.registration import register

register(
    id="object_push-v0",
    entry_point="tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env:ObjectPushEnv",
)