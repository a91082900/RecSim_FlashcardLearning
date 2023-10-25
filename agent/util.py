from recsim.agents import full_slate_q_agent

def create_create_agent(agent=full_slate_q_agent.FullSlateQAgent):
    def create_agent(sess, environment, eval_mode, summary_writer=None):
        kwargs = {
            'observation_space': environment.observation_space,
            'action_space': environment.action_space,
            'summary_writer': summary_writer,
            'eval_mode': eval_mode,
        }
        return agent(sess, **kwargs)
    return create_agent