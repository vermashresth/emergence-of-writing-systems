{
    "agent": "ppo",
    "batch_size": 32,
    "update_frequency": 32,
    "learning_rate": 0.001813150053725916,
    "subsampling_fraction": 0.1,
    "optimization_steps": 50,
    "likelihood_ratio_clipping": 0.09955676846552193,
    "discount": 0.99,
    "critic_network": {"type": "auto", "internal_rnn": false},
    "critic_optimizer": {
        "type": "multi_step",
        "optimizer": {"type": "adam", "learning_rate": 5e-5},
        "num_steps": 5
    },
    "preprocessing": null,
    "exploration": 0.0,
    "variable_noise": 0.0,
    "l2_regularization": 0.0,
    "entropy_regularization": 0.0011393096635237982
}
