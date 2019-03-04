
## Learning Algorithm

The agent is trained with the [DDPG algorithm](https://deepmind.com/research/publications/continuous-control-deep-reinforcement-learning/). The full algorithm is described in the METHODS section of the paper.
- We initialize the replay memory `D` to some capacity `N`.
- We initialize the local Actor and Critic network. The actor network does the policy approximation while the critic does the value estimation.
- We copy those generated weights to the target Actor and target Critic network after every iteration.
- We train the agent for some episodes and for some maximum number of time-steps (`max_t`) in each episode, unless it terminates earlier (e.g. by encountering a terminal state).
- The training loop is composed out of two steps: acting and learning.
- In the acting step, the agent passes the state vector through the Actor network and takes the action which is the output of the network.
- In the learning step, the Critic network is used as a feedback to the Actor network to change its weights such that the estimated value of the input state is maximized.
- Next, we update the *target* Actor and Critic weights by making a copy of the current weights of the local Actor and Critic networks.

**Architecture of Actor Network**

- input size = 33
- output size = 4
- 2 hidden layers and one output layer
- each hidden layer has 128 hidden units and is followed by a `ReLU` activation layer
- We have a batch normalization layer after the first layer
- output layer is followed by a tanh activation layer

**Architecture of Critic Network**

- input size = 4
- output size = 1
- 2 hidden layers and one output layer
- each hidden layer has 128 hidden units and is followed by a `ReLU` activation layer
- We have a batch normalization layer after the first layer
- output layer is followed by a linear activation unit

**Hyperparameters**

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

## Plot of Rewards

![Scores](./images/reward.png)

## Observations/Issues

* Agent seems to learn very slowly initially and then pick up the rewards as we cross the 50th Epoch.

* Using a Batch Normalization layer really helped.

## Ideas for Future Work
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
  - The idea is to implement a Policy Gradient algorithm that determines the appropriate policy with gradient methods. However, the change in the policy from one iteration to another is very slow in the neighbourhood of the previous policy in the high dimensional space.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  - The idea behind using these technique for sampling from the replay buffer is that not all experiences are equal, some are more important than others in terms of reward, so naturally the agent should at least prioritize between the different experiences.
- [Asynchronous Actor Critic](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
  - The idea is to have a global network and multiple agents who all interact with the environment separately and send their gradients to the global network for optimization in an asynchronous way.
