import asyncio
from collections import defaultdict
import random

from strategies.strategy import Strategy, collect_built_in_strategies
from core_types import Task
from storage import get_db
from logs import logger

class Executor:

    def __init__(self) -> None:
        self.strategies: dict[str, type[Strategy]] = {}
        for strategy in collect_built_in_strategies().values():
            self.register_strategy(strategy)

        self.strategies_instances: dict[str, Strategy] = {}

    def register_strategy(self, strategy: type[Strategy]):
        if strategy.NAME in self.strategies:
            raise ValueError(f"Strategy {strategy.NAME} already registered")

        self.strategies[strategy.NAME] = strategy

    async def main(self):
        # 1. Pulls tasks from the database
        # 2. starts worker for each type to batch process them
        #     - priotitizes:
        #          - Tasks with large strategy.priority
        #          - Tasks that have been waiting for a long time
        #          - Tasks that can be processed in parallel efficiently (enough for strategy.max_batch_size)
        #     - does not start strategies that use conflicting resources

        # Okay, simpler for now:
        # 1. Pulls tasks from the database
        # 2. Pick the single highest priority task, if tied, pick the oldest
        # 3. Process the task
        # 4. Repeat

        db = get_db()

        while True:
            tasks = db.get_pending_tasks()
            logger.debug(f"Found {len(tasks)} tasks to run")
            if not tasks:
                await asyncio.sleep(1)
                continue

            tasks_to_run = self.pick_tasks_to_run(tasks)
            await self.run_tasks(tasks_to_run)


            await asyncio.sleep(0.3)

    def pick_tasks_to_run(self, tasks: list[Task]) -> list[Task]:
        assert tasks


        return [random.choice(tasks)]

        tasks_by_strategy = defaultdict(list)
        for task in tasks:
            tasks_by_strategy[task.strategy].append(task)

        # Pick the highest priority strategy, if tied, pick the oldest
        def score(strategy: str):
            return (-self.strategies[strategy].PRIORITY, min(task.created_at for task in tasks_by_strategy[strategy]))


        raise NotImplementedError("Not implemented yet")

    async def run_tasks(self, tasks: list[Task]) -> None:
        if not tasks:
            return

        # All the tasks should have the same strategy
        strategies = set(task.strategy for task in tasks)
        assert len(strategies) == 1, f"Tasks have different strategies: {strategies}"
        strategy = self.strategies[strategies.pop()]

        if strategy.NAME not in self.strategies_instances:
            self.strategies_instances[strategy.NAME] = strategy()

        strategy_instance = self.strategies_instances[strategy.NAME]
        await strategy_instance.process_all(tasks)
