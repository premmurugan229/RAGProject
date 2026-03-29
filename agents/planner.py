from typing import List
from dataclasses import dataclass

@dataclass
class Task:
    type: str
    description: str
    entity: str = None

class PlannerAgent:
    def __init__(self, config):
        from transformers import pipeline

        self.pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512
        )

    def plan(self, query):
        response = self.pipe(f"Break this into tasks: {query}")
        return [{"task": response[0]["generated_text"]}]
