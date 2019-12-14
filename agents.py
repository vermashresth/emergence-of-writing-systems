import json
from tensorforce.agents import Agent

def get_agents(ppo_config, network_config, sender_type, n_features, img_dim, img_features_len, n_samples, action_size):

    with open(ppo_config) as fp:
        agent_spec_big = json.load(fp)

    with open(network_config) as fp:
        network_deep = json.load(fp)

    if sender_type=="agnostic":
        speaker_state_shape = (n_features+img_dim*img_dim,)
    elif sender_type=="aware":
        speaker_state_shape = (n_samples*n_features+img_dim*img_dim,)

    Speaker = Agent.from_spec(
        spec=agent_spec_big,
        kwargs=dict(
            states=dict(type='float', shape=speaker_state_shape),
            actions=dict(type='int', num_actions=img_dim, shape=(action_size,)),
            network=network_deep,
        )
    )

    Listener = Agent.from_spec(
        spec=agent_spec_big,
        kwargs=dict(
            states=dict(type='float', shape=(n_samples*n_features + img_features_len,)),
            actions=dict(type='int', num_actions=n_samples),
            network=network_deep,
        )
    )
    Speaker.reset()
    Listener.reset()

    return Speaker, Listener
