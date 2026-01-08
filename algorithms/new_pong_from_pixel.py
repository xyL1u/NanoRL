import numpy as np
import matplotlib as plt
import pickle
import gymnasium as gym
import ale_py
import shimmy
from gymnasium import pprint_registry
pprint_registry()

# --- Hyperparameter ---
gamma = 0.99 # Discount factor for reward
H = 200 # Number of hidden layer neurons
decay_rate = 0.99 # Discount factor for RMSProp
resume = False # Resume from previous checkpoint?
learning_rate = 1e-4
batch_size = 10

# --- Model Initialization ---
D = 80 * 80

save_file = 'save.p'
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    # Xavier initialization
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# Update buffers that add up gradients over a batch
grad_buffer = {k : np.zeros_like(v) for k,v in model.items()}
# RMSProp memory
rmsprop_cache = {k : np.zeros_like(v) for k,v in model.items()}

# ---------------------------------------------------------------
# Phase 1: Utilities
# ---------------------------------------------------------------

def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(- x))
    return y

def prepro(I):
    
    """
    Preprocess 210x160x3 frame into 80x80 1D float vector
    I: Image
    """
    
    # 1.Crop the image(remove the scoreboard at the top)
    I = I[35:195]

    # 2.Downsampling by factor of 2(take every 2nd pixel)
    I = I[::2, ::2, 0]

    # 3.Erase background
    I[I == 144] = 0
    I[I == 109] = 0

    # 4.Set everything else to 1
    I[I != 0] = 1

    # 5.Flatten to 1D vector
    return I.astype(float).ravel()

def policy_forward(x):
    """
    Forward pass of the policy network.
    Input: x(preprocessed image vector)
    Output:p(probability of moving UP), h(hidden state)
    """
    # 如果我想输出上下左右的概率呢？（cross-entropy）
    # 1. Matrix multiplication for first hidden layer
    h = np.dot(model['W1'], x)

    # 2.ReLU: Set values < 0 to 0
    h[h < 0] = 0

    # 3.Matrix multiplication for output
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)

    return p, h

# ------------- Important!!! ------------
def policy_backward(eph, epdlogp, epx):
    """
    Backward pass. 
    
    Arguments:
    eph -- stack of hidden states (episode_num, H) -> (episode_num, 200)
    epdlogp -- stack of gradients from the output (episode_num, 1) -> "Fake Label - Probability" * "Advantage"
    """

    # Compute gradient of W2 (dW2)
    dW2 = np.dot(eph.T, epdlogp).ravel()

    # Compute gradient of hidden layer (dh)
    dh = np.outer(epdlogp, model['W2'])

    # Backprop through ReLU
    dh[eph <= 0] = 0

    # Compute gradient of W1 (dW1) 
    dW1 = np.dot(dh.T, epx)

    return {'W1': dW1, 'W2': dW2}


def discount_rewards(r):
    """
    Take 1D float array of rewards and compute discounted reward.
    """
    discounted_r = np.zeros_like(r)
    running_add = 0 # running_add refer to the "Value" of the state at time t

    # Iterate backwards through the rewards
    for t in reversed(range(0, r.size)):
        # Pong specific: reset the sum if the reward is non-zero
        # (This indicates a point was scored, ending the short-term interaction)
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]

        discounted_r[t] = running_add

    return discounted_r

# ---------------------------------------------------------------
# Phase 2: Main Loop
# ---------------------------------------------------------------

# Initialize env
env = gym.make('PongNoFrameskip-v4', render_mode='human')
observation, info = env.reset()

pre_obs = None
# obs, hidden_state, rs, grad_logp: observations, hidden layer value, rewards, gradient of log-probabilities
diff_observations, hidden_state, rs, grad_logp = [], [], [], []
reward_running = None
reward_sum = 0
episode = 0
reward_history = []

print('-----Training started-----')

while True:
    # ---------------------------------------------------------------
    # Step 1: Preprocess the "Motion"
    # ---------------------------------------------------------------

    # Preprocess the current observation
    cur_obs = prepro(observation)

    # Calculate the difference frame
    diff_obs = cur_obs - pre_obs if pre_obs is not None else np.zeros(D)
    cur_obs = pre_obs

    # -----------------------------------------------------------------
    # STEP 2: Forward Pass & Action Selection
    # -----------------------------------------------------------------
    
    # up_prob: probability of moving UP(Action 2)
    up_prob, h = policy_forward(diff_obs)
    action = 2 if np.random.uniform() < up_prob else 3

    # -----------------------------------------------------------------
    # STEP 3: Record History (Crucial for Training!)
    # -----------------------------------------------------------------
    
    # Store to calculate gradients late
    diff_observations.append(diff_obs)
    hidden_state.append(h)

    # TODO: Create a "Fake Label" (y)
    # Logic: In supervised learning, y is the truth. 
    # Here, we treat the action we just took as the "truth" for now.
    # If action == 2 (UP), y = 1. If action == 3 (DOWN), y = 0.
    y = 1 if action == 2 else 0

    grad_logp.append(y - up_prob)

    # -----------------------------------------------------------------
    # STEP 4: Step the Environment
    # -----------------------------------------------------------------

    # Execute action
    observation, reward, terminated, truncated, info = env.step(action)

    # Accumulate reward
    reward_sum += reward

    # Record reward
    rs.append(reward)

    # Check if game is over
    if terminated or truncated:
        episode += 1
        reward_history.append(reward_sum)

        # 1. Stack all lists into NumPy arrays
        epx = np.vstack(diff_observations) # (N, 6400)
        eph = np.vstack(hidden_state) # (N, 200)
        epdlogp = np.vstack(grad_logp) # (N, 1)
        epr = np.vstack(rs) # (N, 1)

        # -------------------------------------------------------------
        # Phase 3: Update weights
        # -------------------------------------------------------------
        print(f"----- Episode {episode} finished. Total Reward: {reward_sum} -----")

        # Reset everything for the next episode
        observation, info = env.reset()
        reward_sum = 0
        pre_obs = None
        diff_observations, rs, grad_logp, hidden_state = [], [], [], []
        
        # 2. Compute the 'true_value' of each action by propagating rewards backwards
        discounted_epr = discount_rewards(epr)

        # 3. Normalize Rewards (Crucial Step!)
        # We subtract the mean and divide by the std deviation.
        # This makes the rewards roughly Gaussian (mean 0, std 1).
        # It ensures that "good" actions get positive signal and "bad" actions get negative,
        # preventing the gradients from varying too wildly (Gradient Variance Reduction).
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # 4. Modulate the gradient with Advantage
        # Logic: Policy Gradient = (PseudoLabel - Probability) * Advantage
        # If Advantage (discounted_epr) is positive, we encourage this action.
        # If Advantage is negative, we discourage it.
        epdlogp *= discounted_epr

        # 5. Backward Pass
        grad = policy_backward(eph, epdlogp, epx)

        # 6. Accumulate gradients over the batch
        for k in model:
            grad_buffer[k] += grad[k]
        
        # 7. RMSProp Parameter Update
        # Perform the actual weight update every `batch_size` episodes
        if episode % batch_size == 0:
            print(f"----- Updating weights at episode {episode}. Last reward: {reward_sum} -----")

            for k, v in model.items():
                g = grad_buffer[k]

                # RMSProp Memory Update:
                # cache = decay * cache + (1 - decay) * g^2
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2

                # Parameter Update: Policy ascent
                # W_new = W_old + Learning_Rate * Gradient / Sqrt(Cache)
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)

                # Reset the batch gradient buffer
                grad_buffer[k] = np.zeros_like(v)

        if episode % 100 == 0:
            print(f'----- Saving model to {save_file} -----')
            pickle.dump(model, open(save_file, 'wb'))

            plt.figure(figsize=(10, 5))
            plt.plot(reward_history)
            plt.title('Training Progress (Pong)')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.savefig('training_curve.png')
            plt.close()
        
        MAX_EPISODES = 8000
        if episode >= MAX_EPISODES:
            print(f"----- Game End -----")
            pickle.dump(model, open(save_file, 'wb'))
            break