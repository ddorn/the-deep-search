import asyncio
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import random

from config import Config
from sources import BUILT_IN_SOURCES
from strategies import BUILT_IN_STRATEGIES
from strategies.strategy import Source, Strategy
from core_types import Task
from storage import Database, get_db
from logs import logger

class Executor:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sources: dict[str, Source] = {}
        self.strategies: dict[str, Strategy] = {}

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

    async def new_main(self):
        db = get_db()

        # Delete entries that are not in the new config
        self.apply_config_changes(self.config, db)

        # Load all sources
        for name, source_config in self.config.sources.items():
            source_class = BUILT_IN_SOURCES[source_config.type]
            parsed_config = source_class.CONFIG_TYPE.model_validate(source_config.args)
            self.sources[name] = source_class(parsed_config, name)

        # Load all strategies
        for name, strategy_class in BUILT_IN_STRATEGIES.items():
            raw_config = self.config.strategies.get(name, {})
            parsed_config = strategy_class.CONFIG_TYPE.model_validate(raw_config)
            self.strategies[name] = strategy_class(parsed_config)

        db.restart_crashed_tasks()

        # Mount all sources & strategies
        with self.mount_all():

            # Run the main loop
            while True:
                tasks = db.get_pending_tasks()
                logger.debug(f"Found {len(tasks)} tasks to run")
                if not tasks:
                    await asyncio.sleep(1)
                    continue

                tasks_to_run = self.pick_tasks_to_run(tasks)
                await self.run_tasks(tasks_to_run)

                await asyncio.sleep(0.3)

    @contextmanager
    def mount_all(self):
        with ExitStack() as stack:
            for source in self.sources.values():
                stack.enter_context(source.mount())
            for strategy in self.strategies.values():
                stack.enter_context(strategy.mount())

            yield

    def apply_config_changes(self, new_config: Config, db: Database) -> None:
        """
        Apply the changes in the new config to the old config.
        - This removes documents from sources that are not in the new config.
        - Then save the new config to the database.
        """

        old_config = db.get_last_config()
        if old_config is None or old_config == new_config:
            return

        # Remove sources that are not in the new config
        for source_name in set(old_config.sources) - set(new_config.sources):
            for doc in db.get_documents_from_source(source_name):
                db.delete_document(doc.id)

        # Save the new config
        db.save_config(new_config)
