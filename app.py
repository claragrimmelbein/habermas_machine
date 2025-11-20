from fastapi_poe import PoeBot, run
from habermas_machine import machine

class HabermasPoeBot(PoeBot):
    def __init__(self):
        self.hm = machine.HabermasMachine(
            statement_client="POE",  # use your working Poe backend
            reward_client="POE",
        )

    async def get_response(self, message, context):
        opinions = [message["content"]]  # one-turn for example
        winner, _ = self.hm.mediate(opinions)
        yield {"text": winner}

if __name__ == "__main__":
    run(HabermasPoeBot(), port=8080)
