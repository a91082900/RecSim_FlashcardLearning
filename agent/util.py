from recsim.agents import full_slate_q_agent

def create_agent_helper(agent=full_slate_q_agent.FullSlateQAgent, **kwargs):
    def create_agent(sess, environment, eval_mode, summary_writer=None):
        print(f"using {agent.__name__}")
        kwargs['observation_space'] = environment.observation_space
        kwargs['action_space'] = environment.action_space
        kwargs['summary_writer'] = summary_writer
        kwargs['eval_mode'] = eval_mode

        return agent(sess, **kwargs)
    return create_agent