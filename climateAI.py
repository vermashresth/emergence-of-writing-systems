import numpy as np
from tensorforce.agents import Agent, DQNAgent
import json

a = [0, -.2, -.06, -.29, -.13, -.056, -.15]
b = [0, 6, 2.5, 6.28, 6, 3.74, 7.6]
c = [0, -5, 0, -3, -6, -23, -15]
al = [0, 12, 10, 8, 6, 15, 10]

spec = "configs/ppo.json"
network_config = "configs/mlp3_network.json"

def init_agents(n_agents, spec):
    agents = []
    with open(spec) as fp:
        agent_spec = json.load(fp)
    with open(network_config) as fp:
        network_deep = json.load(fp)
    for i in range(n_agents):
        myagent = Agent.from_spec(
            spec=agent_spec,
            kwargs=dict(
                states=dict(type='float', shape=(3,), num_values=3),
                actions=dict(type='int', num_actions=11, min_value=0, max_value=10),
                network=network_deep,
            )
        )
        # network = [
        #     dict(type='dense', size=64, activation='softmax'),
        #         ]
        # myagent = DQNAgent(
        #     states=dict(type='float', shape=(3,)),
        #     actions=dict(type='int', num_actions=11, min_value=0, max_value=10),
        #     network=network,
        #     # update_mode = dict(units='episodes', batch_size=2, frequency=2),
        #     batching_capacity = 256
        # )
        agents.append(myagent)
    return agents

    # agent = Agent.create(
    # agent='tensorforce',
    # states=dict(type='float', shape=(3,)),
    # actions=dict(type='int', num_values=11, min_value=0, max_value=10),
    # max_episode_timesteps=100,
    # memory=10000,
    # update=dict(unit='timesteps', batch_size=64),
    # optimizer=dict(type='adam', learning_rate=3e-4),
    # policy=dict(network='auto'),
    # objective='policy_gradient',
    # reward_estimation=dict(horizon=20)
    # )
    # network = [
    #     dict(type='flatten'),
    #     dict(type='dense', size=64, activation='swish'),
    #     dict(type='dense', size=64, activation='swish')
    #         ]
    #
    #     print("Creating agent.")


class Game:
    def __init__(self,agent_list):
        self.agents = agent_list
        self.reset_game()

    def reset_game(self):
        self.seed = np.random.choice([0,1,2])
        self.Q1 = [160, 115, 80][self.seed]
        self.Q2 = [65, 50, 35][self.seed]
        self.S = [15, 12, 10][self.seed]

    def collect_actions(self):
        self.actions = []
        for agent in self.agents:
            action = agent.act(states = [self.Q1, self.Q2, self.S], independent=False)
            self.actions.append(action)
    def cal_violations(self):
        x1,x2,x3,x4,x5,x6 = self.x
        v = [0,0,0,0,0,0,0,0,0]

        v[0] = al[1]-x1
        v[1] = al[2]-self.Q1+x1
        v[2] = x2 - self.S -self.Q1 +x1
        v[3] = al[4] - x3
        v[4] = al[3] - x4
        v[5] = al[4] - self.Q2 + x4
        v[6] = al[6] - x5
        v[7] = al[5] - x6
        v[8] = al[6] - x2 -x3 + x6

        pen = 0
        n_viol = 0
        for i in range(9):
            if v[i]>0:
                n_viol+=1
                pen += (v[i]+1)*100
        return pen, n_viol
    def distribute_rewards(self, ep):
        batch = ep%64
        if ep==0:
            self.f_b=[]
            self.e_b = []
        rewards = []
        # for i, action in enumerate(self.actions):
        x1, x2, x4, x6 = list(np.array(self.actions)*0.1)
        x1 = al[1]+(self.Q1-al[2]-al[1])*x1
        x4 = al[3]+(self.Q2-al[4]-al[3])*x4
        x2 = (self.S+self.Q1-al[1])*x2
        x6 = al[5]+(self.S+self.Q1+self.Q2-al[1]-al[3]-al[6]-al[5])*x6

        x3 = self.Q2 - x4
        x5 = x3 + x3 - x6
        x = [x1,x2,x3,x4,x5,x6]
        self.x = x
        f = 0
        for j in range(1, 6):
            f += a[j]*x[j]**2+b[j]*x[j]+c[j]
        pen, n_viol = self.cal_violations()
        f-= pen

        if batch==0:
            self.f_b.append(f)
            self.e_b.append(True)
            for agent in self.agents:
                try:
                    agent.model.observe(reward=self.f_b, terminal=self.e_b)
                except:
                    print(batch, ep, self.f_b, self.e_b)
            print(n_viol)
        else:
            self.f_b.append(f)
            self.e_b.append(True)
        if batch==0:
            self.f_b=[]
            self.e_b = []
        return f

episodes = 1000000
n_agents = 4
agents = init_agents(n_agents, spec)
game = Game(agents)
big_rew = []
for ep in range(episodes):
    game.reset_game()
    game.collect_actions()
    rew = game.distribute_rewards(ep)
    big_rew.append(rew)
    big_rew = big_rew[-10000:]
    if ep%1000==0:
        print("episode", ep, " : ", rew, game.seed, game.actions, "avg rew", sum(big_rew)/10000)
        # print("avg rew", sum(big_rew))
