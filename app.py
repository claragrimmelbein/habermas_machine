from fastapi_poe import PoeBot, run
from habermas_machine import machine, types

class HabermasPoeBot(PoeBot):
    def __init__(self):
        # Poe-based language-model clients
        statement_client = types.LLMClient.POE.get_client("Claude-3-Opus")
        reward_client = types.LLMClient.POE.get_client("Claude-3-Opus")

        # Corresponding model task wrappers
        statement_model = types.StatementModel.STATEMENT_GENERATION.get_model(statement_client)
        reward_model = types.RewardModel.REWARD_SCORING.get_model(reward_client)

        question = "How can communities encourage sustainable energy adoption?"

        # Instantiate HabermasMachine with all 6 required positional args
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
