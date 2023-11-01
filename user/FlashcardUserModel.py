from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from .UserState import UserState
from .UserSampler import UserSampler
from .UserResponse import UserResponse
from util import eval_result
import numpy as np

class FlashcardUserModel(user.AbstractUserModel):
  def __init__(self, num_candidates, time_budget, slate_size, 
    eval_delay_time=0, training_param=None, seed=0, sample_seed=0):
    super(FlashcardUserModel, self).__init__(
        UserResponse, UserSampler(
            UserState, num_candidates, time_budget, 
            seed=sample_seed
        ), slate_size)
    self.choice_model = MultinomialLogitChoiceModel({})
    self._rng = np.random.RandomState(seed)
    self._eval_delay_time = eval_delay_time
    self.training_param = training_param

  def is_terminal(self):
    terminated = self._user_state._time >= self._user_state._time_budget
    if terminated: # run evaluation process
      eval_result(self._user_state._time_budget + self._eval_delay_time,
                  self._user_state._last_review.copy(),
                  self._user_state._history.copy(),
                  self._user_state._W.copy(),
                  self.write_results)
    return terminated

  def write_results(self, eval_time, pr, score, filename="result.csv"):
    with open(filename, "a+") as f:
      f.seek(0)
      if f.readline() == '':
        f.write("lr,alpha,T,n,s,eval_time,score,p")
        f.write(',p'.join(map(str, range(1, 31))))
        f.write("\n")
      if self.training_param != None and len(self.training_param) == 2:
        f.write(f"{self.training_param[0]},{self.training_param[1]},")
      else:
        f.write(",,")
      f.write(f"{self._user_state._time_budget},{self._user_state._cards},{self._eval_delay_time},{eval_time},{score},")
      f.write(",".join(map(str, pr)))
      f.write("\n")

  def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
      doc_id = doc._doc_id
      self._user_state._history[doc_id][0] += 1
      if response._recall:
        self._user_state._history[doc_id][1] += 1
      else:
        self._user_state._history[doc_id][2] += 1
      self._user_state._last_review[doc_id] = self._user_state._time
    self._user_state._time += 1

  def simulate_response(self, slate_documents):
    responses = [self._response_model_ctor() for _ in slate_documents]
    # Get click from of choice model.
    self.choice_model.score_documents(
      self._user_state, [doc.create_observation() for doc in slate_documents])
    scores = self.choice_model.scores
    selected_index = self.choice_model.choose_item()
    # Populate clicked item.
    self._generate_response(slate_documents[selected_index],
                            responses[selected_index])
    return responses

  def _generate_response(self, doc, response):
    # W = np.array([1,1,1])
    doc_id = doc._doc_id
    W = self._user_state._W[doc_id]
    if not W.any(): # uninitialzed
      error = self._user_state._doc_error[doc_id] # a uniform error for each user
      self._user_state._W[doc_id] = W = doc.base_difficulty + error
      print(W)
    # use exponential function to simulate whether the user recalls
    last_review = self._user_state._time - self._user_state._last_review[doc_id]
    x = self._user_state._history[doc_id]

    pr = np.exp(-last_review / np.exp(np.dot(W, x))).squeeze()
    print(f"time: {self._user_state._time}, reviewing flashcard {doc_id}, recall rate = {pr}")
    if self._rng.random_sample() < pr: # remembered
      response._recall = True
    response._pr = pr
  
  def recv_docs(self, docs):
    # initialize Ws
    for doc_id in docs:
      did = int(doc_id)
      error = self._user_state._doc_error[did] # a uniform error for each user
      self._user_state._W[did] = docs[doc_id] + error
      print(doc_id, docs[doc_id])