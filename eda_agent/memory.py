class AgentMemory:
    def __init__(self):
        self.conclusions = []

    def add_conclusion(self, conclusion: str):
        self.conclusions.append(conclusion)

    def get_conclusions(self):
        return self.conclusions
