import tensorflow as tf
from recsim.simulator import environment
from user import FlashcardUserModel
from document import FlashcardDocumentSampler
from recsim.simulator import recsim_gym
from recsim.agents import full_slate_q_agent
from recsim.simulator import runner_lib
from agent import create_create_agent
from util import reward, update_metrics

slate_size = 1
num_candidates = 10
time_budget = 60

tf.compat.v1.disable_eager_execution()

create_agent_fn = create_create_agent(full_slate_q_agent.FullSlateQAgent)

ltsenv = environment.Environment(
  FlashcardUserModel(num_candidates, time_budget, slate_size),
  FlashcardDocumentSampler(),
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
    episode_log_file="",
    max_training_steps=5,
    num_iterations=1
)

runner.run_experiment()