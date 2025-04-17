import asyncio
from collections import defaultdict
from contextlib import ExitStack, contextmanager

from config import Config
from core_types import PartialTask, Rule, Task, TaskStatus, TaskType
from logs import logger
from sources import BUILT_IN_SOURCES
from storage import Database
from strategies import BUILT_IN_STRATEGIES
from strategies.strategy import Module


class Executor:

    def __init__(self, db: Database) -> None:
        self.strategies: dict[str, Module] = {}
        self.rules: list[Rule] = []
        self.db = db

    async def main(self):

        self.db.check_and_clean_database(prompt_user=True)

        self.db.delete_tasks_that_should_be_deleted_at_shutdown()
        self.db.restart_in_progress_tasks()
        self.db.commit()

        # Delete entries that are not in the new config
        self.apply_config_changes(self.db.config)

        # Load all strategies
        for name, strategy_class in BUILT_IN_STRATEGIES.items():
            raw_config = self.db.config.strategies.get(name, {})
            parsed_config = strategy_class.CONFIG_TYPE.model_validate(raw_config)
            self.strategies[name] = strategy_class(parsed_config, self.db)
            self.rules = self.strategies[name].add_rules(self.rules)

        # Load all sources
        for name, source_config in self.db.config.sources.items():
            source_class = BUILT_IN_SOURCES[source_config.type]
            parsed_config = source_class.CONFIG_TYPE.model_validate(source_config.args)
            assert name not in self.strategies, f"There is already a strategy named '{name}'. Use a different name for the source."
            self.strategies[name] = source_class(parsed_config, self.db, name)
            self.rules = self.strategies[name].add_rules(self.rules)

        # Mount all sources & strategies
        with self.mount_all():
            self.db.commit()

            # Run the main loop
            while True:
                self.create_tasks_from_unhandled_assets()
                tasks = self.db.get_pending_tasks()
                logger.debug(f"Found {len(tasks)} tasks to run")
                if not tasks:
                    await asyncio.sleep(1)
                    continue

                tasks_to_run = self.pick_tasks_to_run(tasks)
                if not tasks_to_run:
                    await asyncio.sleep(1)
                    continue

                await self.run_tasks(tasks_to_run)
                self.db.commit()

                await asyncio.sleep(0.1)

    def pick_tasks_to_run(self, tasks: list[Task]) -> list[Task]:
        assert tasks

        tasks_by_strategy = defaultdict(list)
        for task in tasks:
            tasks_by_strategy[task.strategy].append(task)

        # Pick the highest priority strategy
        strategy = max(tasks_by_strategy.keys(), key=self.strategy_priority)
        if strategy in self.db.config.global_config.paused_strategies:
            logger.debug(f"Strategy '{strategy}' is paused, skipping")
            return []

        tasks_for_best_strategy = tasks_by_strategy[strategy]
        batch_to_run = tasks_for_best_strategy[: self.strategies[strategy].MAX_BATCH_SIZE]
        return batch_to_run

    def strategy_priority(self, strategy: str) -> int:
        if strategy in self.db.config.global_config.paused_strategies:
            return -9999
        return self.strategies[strategy].PRIORITY

    async def run_tasks(self, tasks: list[Task]) -> None:
        if not tasks:
            return

        # All the tasks should have the same strategy
        strategies = set(task.strategy for task in tasks)
        assert len(strategies) == 1, f"Tasks have different strategies: {strategies}"
        strategy = self.strategies[strategies.pop()]

        logger.debug(f"Running {len(tasks)} tasks for strategy '{tasks[0].strategy}'")
        task_ids = [task.id for task in tasks]
        self.db.set_task_status(TaskStatus.IN_PROGRESS, task_ids)
        await strategy.process_all(tasks)

        to_delete = []
        to_done = []
        for task in tasks:
            if task.task_type in [TaskType.DELETE_ONCE_RUN, TaskType.DELETE_AT_SHUTDOWN]:
                to_delete.append(task.id)
            else:
                to_done.append(task.id)

        self.db.set_task_status(TaskStatus.DONE, to_done)
        self.db.delete_tasks(to_delete)


    @contextmanager
    def mount_all(self):
        with ExitStack() as stack:
            for strategy in self.strategies.values():
                stack.enter_context(strategy.mount())

            yield

    def apply_config_changes(self, new_config: Config) -> None:
        """
        Apply the changes in the new config to the old config.
        - This removes documents from sources that are not in the new config.
        - Then save the new config to the database.
        """

        old_config = self.db.get_last_config()
        if old_config is None or old_config == new_config:
            return

        # Remove modules that are not in the new config
        # for strategy_name in set(old_config.strategies) - set(new_config.strategies):
        #     for doc in self.db.get_documents_from_source(strategy_name):
        #         self.db.delete_document(doc.id)
        # TODO: also delete their folder

        # Save the new config
        self.db.save_config(new_config)

    def create_tasks_from_unhandled_assets(self):
        assets = self.db.get_unhandled_assets()
        documents = self.db.get_documents([asset.document_id for asset in assets])

        for asset, document in zip(assets, documents, strict=True):
            for rule in self.rules:
                if rule.matches(asset, document):
                    self.db.create_task(
                        PartialTask(
                            strategy=rule.strategy,
                            document_id=asset.document_id,
                            input_asset_id=asset.id,
                            status=TaskStatus.PENDING,
                        )
                    )

            self.db.set_asset_next_steps_created(asset.id)
