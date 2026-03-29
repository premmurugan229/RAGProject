from transformers import pipeline

class OrchestratorAgent:
    def __init__(self, config):
        self.pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512
        )
       
    def generate(self, context: str, query: str) -> str:
        return self.pipe(query)[0]["generated_text"]
