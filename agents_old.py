import json
from tensorforce.agents import Agent

def get_agents(ppo_config, network_config, sender_type, n_features, img_dim, img_features_len, n_samples, num_actions, action_size, n_pop=0):

    with open(ppo_config) as fp:
        agent_spec_big = json.load(fp)

    with open(network_config) as fp:
        network_deep = json.load(fp)

    if sender_type=="agnostic":
        speaker_state_shape = (n_features+img_dim*img_dim+n_pop,)
    elif sender_type=="aware":
        speaker_state_shape = (n_samples*n_features+img_dim*img_dim+n_pop,)

    Speaker = Agent.from_spec(
        spec=agent_spec_big,
        kwargs=dict(
            states=dict(type='float', shape=speaker_state_shape),
            actions=dict(type='int', num_actions=num_actions, shape=(action_size,)),
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

def get_symbol_agents(n_features, n_samples, action_size):

    if sender_type=="agnostic":
        speaker_state_shape = (n_features,)
    elif sender_type=="aware":
        speaker_state_shape = (n_samples*n_features,)

    # Instantiate a Tensorforce agent
    speaker = PPOAgent(
        states=dict(type='float', shape=(n_features,)),
        actions=dict(type='int', num_actions=n_vocab),
        network=[
            dict(type='dense', size=64),
            dict(type='dense', size=128),
            dict(type='dense', size=64),
        ],
        step_optimizer=dict(type='adam', learning_rate=1e-4)
    )

    listener = PPOAgent(
        states=dict(type='float', shape=(n_samples*n_features + 1,)),
        actions=dict(type='int', num_actions=n_samples),
        network=[
            dict(type='dense', size=64),
            dict(type='dense', size=128),
            dict(type='dense', size=64),
        ],
        step_optimizer=dict(type='adam', learning_rate=1e-4)
    )
