from fastapi_poe import PoeBot, run
from habermas_machine import machine, types

class HabermasPoeBot(PoeBot):
    def __init__(self):
        statement_client = types.LLMClient.POE.get_client("Claude-3-Opus")
        reward_client = types.LLMClient.POE.get_client("Claude-3-Opus")

        question = "How can communities encourage sustainable energy adoption?"

        self.hm = machine.HabermasMachine(
            question=question,
            statement_model=statement_client,
            reward_model=reward_client,
            social_choice_method=types.SocialChoiceMethod.MAJORITY_VOTE,
        )

    async def get_response(self, message, context):
        opinions = [message["content"]]
        winner, _ = self.hm.mediate(opinions)
        yield {"text": winner}

if __name__ == "__main__":
    run(HabermasPoeBot(), port=8080)
