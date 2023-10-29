from recsim.simulator import environment
import collections

class DocAccessibleEnvironment(environment.SingleUserEnvironment):
    def __init__(self,
               user_model,
               document_sampler,
               num_candidates,
               slate_size,
               resample_documents=False):
        super(DocAccessibleEnvironment, self).__init__(
               user_model,
               document_sampler,
               num_candidates,
               slate_size,
               resample_documents)
        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation())
        self.pass_documents(self._current_documents)

    def pass_documents(self, docs):
        self._user_model.recv_docs(docs)

    def reset(self):
        self._user_model.reset()
        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation())
        self.pass_documents(self._current_documents)
        user_obs = self._user_model.create_observation()
        return (user_obs, self._current_documents)