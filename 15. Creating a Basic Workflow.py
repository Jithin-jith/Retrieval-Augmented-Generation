from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
import asyncio
class MyWorkflow(Workflow):
    @step
    def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent("Hello, world!")

w = MyWorkflow(timeout=10, verbose=True)
result = w.my_step(StartEvent())
print(result)   

