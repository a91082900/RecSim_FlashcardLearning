from recsim.agent import AbstractEpisodicRecommenderAgent
import tensorflow as tf
import numpy as np

class UCBAgent(AbstractEpisodicRecommenderAgent):
  def __init__(self, sess, observation_space, action_space, eval_mode, 
      eval_delay_time=0, alpha=1.0, learning_rate=0.001, summary_writer=None):
    super(UCBAgent, self).__init__(action_space, summary_writer)
    self._num_candidates = int(action_space.nvec[0])
    self._W = tf.Variable(np.random.uniform(0, 5, size=(self._num_candidates, 3)), name='W')
    self._sess = sess
    self._return_idx = None
    self._prev_pred_pr = None
    self._opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    self._alpha = alpha
    self._deadline = None
    self._eval_delay_time = eval_delay_time # eval at T + s
    self._history_x = [] # list of (time_since_last_review, history)
    self._history_y = []

    assert self._slate_size == 1
  def begin_episode(self, observation=None):
    docs = observation['doc']
    user = observation['user']

    self._deadline = user['time_budget']
    self._history_x = [([], []) for _ in range(len(docs))]
    self._history_y = [[] for _ in range(len(docs))]

    if 'W' in user:
      assign = self._W.assign(user['W'])
      self._sess.run(assign)
    else:
      w = []
      for doc_id in docs:
        w.append(docs[doc_id])
      w = np.array(w).reshape((-1, 3))
      print("observe from docs")
      assign = self._W.assign(w)
      self._sess.run(assign)
      print(self._W.eval(session=self._sess))

    self._episode_num += 1
    return self.step(0, observation)
  def step(self, reward, observation):
    docs = observation['doc']
    user = observation['user']
    response = observation['response']

    if self._return_idx != None and response != None:
      # update w
      self._history_y[self._return_idx].append(response[0]['recall'])
      y_true = self._history_y[self._return_idx]
      # use same W to predict all results
      time_since_last_review, history = self._history_x[self._return_idx]
      history = np.array(history)
      wi = self._W[self._return_idx]

      print("history =", history)
      print("y_history =", y_true)
      mem_param = tf.math.exp(tf.reduce_sum(history * wi, axis=1))
      y_pred = tf.math.exp(-(time_since_last_review / mem_param))
      loss = tf.losses.binary_crossentropy(y_true, y_pred)
      self._sess.run(self._opt.minimize(loss))
      print("y_pred:", y_pred.eval(session=self._sess))
      print("loss:", loss.eval(session=self._sess))
      print("w:", self._W.eval(session=self._sess))
    # base_pr = self.calc_prs(user['time'], user['last_review'], user['history'], self._W)

    time = user['time'] + 1
    history_pos = user['history'].copy()
    history_pos[:, [0, 1]] += 1 # add n, n+ by 1
    history_neg = user['history'].copy()
    history_neg[:, [0, 2]] += 1 # add n, n- by 1
    last_review_now = np.repeat(user['time'], len(user['last_review']))

    # always evaluate at deadline + eval_delay_time
    eval_time = self._deadline + self._eval_delay_time
    pr = self.calc_prs(eval_time, user['last_review'], user['history'], self._W)
    pr_pos = self.calc_prs(eval_time, last_review_now, history_pos, self._W)
    pr_neg = self.calc_prs(eval_time, last_review_now, history_neg, self._W)

    gain = (pr_pos + pr_neg) / 2 - pr
    time_since_last_review = user['time'] - user['last_review']
    uncertainty = self._alpha * tf.math.sqrt(tf.math.log(time_since_last_review) / user['history'][:, 0])
    # print(gain.eval(session=self._sess))
    # print(time_since_last_review)
    # print(uncertainty.eval(session=self._sess))
    ucb_score = gain + uncertainty
    print("       gain:", gain.eval(session=self._sess))
    print("uncertainty:", uncertainty.eval(session=self._sess))
    best_idx = tf.argmax(ucb_score)

    self._return_idx = self._sess.run(best_idx)
    # save history
    self._history_x[self._return_idx][0].append(user['time'] - user['last_review'][self._return_idx])
    self._history_x[self._return_idx][1].append(user['history'][self._return_idx].copy())
    return [self._return_idx]

    
  def calc_prs(self, train_time, last_review, history, W):
    last_review = train_time - last_review
    mem_param = tf.math.exp(tf.reduce_sum(history * W, axis=1))
    pr = tf.math.exp(-last_review / mem_param)
    return pr