import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from game import Game
    from data_preprocess import get_data_points
    from utils import get_feature_extractor_model, get_features, message_to_image, message_to_image_sm
    import numpy as np

    import json
    from tensorforce.agents import Agent
    # import matplotlib
    # matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt

# %matplotlib inline
n_all_features = 512

n_clusters = 10
n_samples = 3
n_vocab = 3
max_len = 2

img_dim = 10
img_rows = 105
img_cols = 105
img_features_len = 32
n_batches = 32
n_epochs = 10000000
action_size = 6
num_actins = img_dim

ppo_config = "configs/ppo-big-200.json"
network_config = "configs/mlp3_network-200.json"
sender_type = "aware"
data_csv_path = "data_csvs/cifar_feats.csv"
feat_model_path = 'feature_extractors/feature_model.h5'
feature_model_json_path = 'feature_extractors/feature_model.json'
flip_s = False
flip_l = False
freq = 1
debug = True

class playGame:
    def __init__(self):


        self.X1, self.Y1, n_features = get_data_points(data_csv_path, n_all_features)
        self.Speaker, self.Listener = self.get_agents(ppo_config, network_config, sender_type, n_features, img_dim, img_features_len, n_samples, num_actions, action_size)
        self.intermediate_layer_model = get_feature_extractor_model(feat_model_path, feature_model_json_path)

        self.big_rewards = []
        self.game_pool = []
        for i in range(n_batches):
            self.game_pool.append(Game(n_features, n_clusters, n_samples, n_vocab, sender_type, [self.X1, self.Y1]))


        self.data = []



    def get_agents(self, ppo_config, network_config, sender_type, n_features, img_dim, img_features_len, n_samples, num_actions, action_size):

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

    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()

        fig.show()
        fig.canvas.draw()
        for i in range(n_epochs):

            message_batch = []
            img_batch = []
            r = i%(2*freq)
            if r<1*freq:
                flip_s = False
                flip_l = True
            if r>=1*freq:
                flip_s = True
                flip_l = False
            l_rewards = []
            strokes = []
            for b in range(n_batches):
                strokes = []
                for idx in range(2):
                    if idx ==0:
                        feat = np.zeros((10,10), np.uint8)
                    else:
                        im = message_to_image_sm(strokes, img_dim, img_dim)
        #                 feat = get_features([im])[0]
                        feat = im.flatten()
                    message = self.Speaker.act(states= self.game_pool[b].speaker_input(feat), independent=flip_s)  # (scalar between 0 and 4)
                    message_batch.append(message)
                    strokes.append(message)
                img = message_to_image(strokes, img_rows, img_cols)
                img_batch.append(img)
        #         Speaker.reset()

            features_batch = get_features(self.intermediate_layer_model, img_batch)

            out_batch = []
            reward_batch = []
            for b in range(n_batches):
                out = self.Listener.act(states = self.game_pool[b].listener_input(features_batch[b]), independent = flip_l)
                out_batch.append(out)
                rew = self.game_pool[b].reward(out)
                reward_batch.append(rew)
                self.game_pool[b].reset()
            speaker_terminals = []
            rew_speaker = []
            for rew in reward_batch:
                speaker_terminals.extend([False, True])
                rew_speaker.extend([0, rew])

            if not flip_s:
        #         print('e1')
                e1 = self.Speaker.model.observe(reward=rew_speaker, terminal=speaker_terminals)
            if not flip_l:
        #         print('e2')
                e2 = self.Listener.model.observe(reward=reward_batch, terminal=[True]*n_batches)
        #     assert e1 == e2
        #     e = e1
            self.big_rewards.extend(reward_batch)
            self.big_rewards = self.big_rewards[-600:]
            if i%(100)==0:
        #         print(" output: ", out)
                avg_rew = sum(self.big_rewards)/len(self.big_rewards)
                print(" message: ", message, " output: ", out, " epoch no: ", i, " avg reward", avg_rew)
                print(" epoch no: ", i, " avg reward", avg_rew)

                if(i!=0):
                    self.data.append(avg_rew)
                ax.clear()
                ax.plot(self.data)
                if not flip_s:
                    ax.set_xlabel('speaker')
                else:
                    ax.set_xlabel('listener')
                fig.canvas.draw()


        #     if sum(self.big_rewards)/len(self.big_rewards)>.75:
        #         break
# pg = playGame()
# pg.run()
