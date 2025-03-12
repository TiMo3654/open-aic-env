from gymnasium.envs.registration import register

register(
    id="floor_env/FloorEnv-v0",
    entry_point="floor_env.envs:FloorEnv",
)

# register(
#     id="floor_env_seq/FloorEnvSeq-v99",
#     entry_point="floor_env.envs:FloorEnvSeq",
# )

register(
    id="floor_env_seq_v0/FloorEnvSeq-v0",
    entry_point="floor_env.envs:FloorEnvSeqV0",
)

register(
    id="floor_env_seq_v1/FloorEnvSeq-v1",
    entry_point="floor_env.envs:FloorEnvSeqV1",
)

register(
    id="floor_env_seq_v2/FloorEnvSeq-v2",
    entry_point="floor_env.envs:FloorEnvSeqV2",
)

register(
    id="floor_env_seq_v3/FloorEnvSeq-v3",
    entry_point="floor_env.envs:FloorEnvSeqV3",
)

register(
    id="floor_env_seq_v4/FloorEnvSeq-v4",
    entry_point="floor_env.envs:FloorEnvSeqV4",
)

register(
    id="floor_env_seq_v5/FloorEnvSeq-v5",
    entry_point="floor_env.envs:FloorEnvSeqV5",
)

register(
    id="floor_env_seq_v6/FloorEnvSeq-v6",
    entry_point="floor_env.envs:FloorEnvSeqV6",
)

register(
    id="floor_env_seq_v7/FloorEnvSeq-v7",
    entry_point="floor_env.envs:FloorEnvSeqV7",
)

register(
    id="floor_env_seq_v8/FloorEnvSeq-v8",
    entry_point="floor_env.envs:FloorEnvSeqV8",
)

register(
    id="floor_env_seq_v9/FloorEnvSeq-v9",
    entry_point="floor_env.envs:FloorEnvSeqV9",
)
