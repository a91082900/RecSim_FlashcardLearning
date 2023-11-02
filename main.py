import tensorflow as tf
# from recsim.simulator import environment
from environment import environment
from user import FlashcardUserModel
from document import FlashcardDocumentSampler
from recsim.simulator import recsim_gym
from recsim.agents import full_slate_q_agent
from recsim.simulator import runner_lib
from agent import *
from util import reward, update_metrics
import numpy as np
import os

np.set_printoptions(suppress=True)

slate_size = 1
num_candidates = 20
time_budget = 50
eval_delay_time = 30

tf.compat.v1.disable_eager_execution()

lr_range = (0.0001, 1)
alpha_range = (0.0001, 10)

log_lr_range = np.log(lr_range)
log_alpha_range = np.log(alpha_range)
log_lrs = np.arange(log_lr_range[0], log_lr_range[1]+1e-3, 
  (log_lr_range[1]-log_lr_range[0])/20)
log_alphas = np.arange(log_alpha_range[0], log_alpha_range[1]+1e-3, 
  (log_alpha_range[1]-log_alpha_range[0])/20)

for i in range(len(log_alphas)):
  print(f"trial {i:02d}")
  lr = 0.01 #np.exp(np.random.uniform(*log_lr_range))
  alpha = np.exp(log_alphas[i])

  create_agent_fn = create_agent_helper(UCBAgent, 
    alpha=alpha, learning_rate=lr, eval_delay_time=eval_delay_time)
  ltsenv = environment.DocAccessibleEnvironment(
    FlashcardUserModel(num_candidates, time_budget, 
      slate_size, eval_delay_time=eval_delay_time, 
      training_param=(lr, alpha), seed=0, sample_seed=0),
    FlashcardDocumentSampler(seed=0),
    num_candidates,
    slate_size,
    resample_documents=False)

  lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, reward, update_metrics)
  lts_gym_env.reset()

  tmp_base_dir = './recsim/'
  runner = runner_lib.TrainRunner(
      base_dir=tmp_base_dir,
      create_agent_fn=create_agent_fn,
      env=lts_gym_env,
      episode_log_file="episode.log",
      max_training_steps=5,
      num_iterations=1
  )

  runner.run_experiment()
  os.system("rm recsim -r")