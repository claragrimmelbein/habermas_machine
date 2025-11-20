from fastapi_poe import PoeBot, run
from habermas_machine import machine, types

class HabermasPoeBot(PoeBot):
    def __init__(self):
        # Poe-based LLM Clients
        statement_client = types.LLMCLient.POE.get_client("Claude-3-Opus")
        reward_client    = types.LLMCLient.POE.get_client("Claude-3-Opus")

        # Create model wrappers that use the clients
        statement_model = types.StatementModel.CHAIN_OF_THOUGHT.get_model()
        reward_model = types.RewardModel.CHAIN_OF_THOUGHT_RANKING.get_model()

        # The deliberation question / topic
        question = "How can communities encourage sustainable energy adoption?"

        # Instantiate the Habermas Machine (notice we pass *all six* args)
        self.hm = machine.HabermasMachine(
            question=question,
            statement_client=statement_client,
            reward_client=reward_client,
            statement_model=statement_model,
            reward_model=reward_model,
            social_choice_method=types.SocialChoiceMethod.MAJORITY_VOTE,
        )

    async def get_response(self, message, context):
        opinions = [message["content"]]
        winner, _ = self.hm.mediate(opinions)
        yield {"text": winner}

if __name__ == "__main__":
    run(HabermasPoeBot(), port=8080)
