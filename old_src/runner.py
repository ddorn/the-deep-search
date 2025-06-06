import threading
import traceback
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable

from rich.progress import (
    BarColumn,
    Column,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class Result:
    is_error: int

    def unwrap(self):
        try:
            return self._result
        except AttributeError:
            raise RuntimeError("No result in this result")

    def unwrap_error(self) -> BaseException:
        try:
            return self._error
        except AttributeError:
            raise RuntimeError("No error in this result")

    def unwrap_or(self, default):
        try:
            return self._result
        except AttributeError:
            return default

    def unwrap_error_or(self, default):
        try:
            return self._error
        except AttributeError:
            return default


class ErrorResult(Result):
    is_error = True

    def __init__(self, error: Exception):
        self._error = error

    def __repr__(self):
        return f"ErrorResult({self._error})"


class SuccessResult(Result):
    is_error = False

    def __init__(self, result: Any):
        self._result = result

    def __repr__(self):
        return f"SuccessResult({self._result})"


@dataclass
class Task:
    func: Callable
    args: tuple
    kwargs: dict
    name: str

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


class Parallel:
    def __init__(
        self,
        n_jobs: int = 1,
        progress_per_task: bool = False,
        show_done: bool = True,
        show_threads: bool = True,
    ):
        self.n_jobs = n_jobs
        self.tasks: list[Task] = []
        self.progress_per_task = progress_per_task
        self.results: dict[int, Result] = {}
        self.to_process = Queue[int]()

        self.show_done = show_done
        self.show_threads = show_threads
        self.progress_bars: list[TaskID] = []
        self.progress = self.mk_progress()

    def mk_progress(self) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}", table_column=Column(ratio=1)),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            BarColumn(30),
            expand=True,
        )

    def add[**P](self, func: Callable[P, Any], title: str = "task") -> Callable[P, None]:
        def task_adder(*args, **kwargs):
            self.tasks.append(Task(func, args, kwargs, title))
            self.to_process.put(len(self.tasks) - 1)

        return task_adder

    def run(self) -> list[Result]:

        with self.progress:
            self.setup_progress()

            if self.n_jobs == 1:
                self.thread_runner(0)
            else:
                threads = []
                for runner_id in range(self.n_jobs):
                    thread = threading.Thread(target=self.thread_runner, args=(runner_id,))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            self.report_all_end()

        return [self.results[i] for i in range(len(self.tasks))]

    def thread_runner(self, runner_id: int) -> None:

        try:
            while not self.to_process.empty():
                task_id = self.to_process.get()
                task = self.tasks[task_id]

                self.report_start_task(runner_id, task_id)
                try:
                    result = task()
                    self.set_result(task_id, SuccessResult(result))
                except Exception as e:
                    self.set_result(task_id, ErrorResult(e))
                    self.report_task_error(runner_id, task_id)
                else:
                    self.report_task_success(runner_id, task_id)
                finally:
                    self.report_task_end(runner_id, task_id)

            self.report_thread_end(runner_id)
        except BaseException:
            self.report_thread_error(runner_id)
            raise

    def set_result(self, task_id: int, result: Result):
        self.results[task_id] = result

    def setup_progress(self):
        if self.show_threads:
            self.progress_bars = [
                self.progress.add_task(f"{i} Thread started", total=None)
                for i in range(self.n_jobs)
            ]
        self.progress_bars.append(
            self.progress.add_task("[green]Total progress", total=len(self.tasks))
        )

    def report_start_task(self, runner_id: int, task_id: int):
        if self.show_threads:
            self.update_thread_progress(
                runner_id,
                description=f"[{runner_id}] Processing [yellow]{self.tasks[task_id].name}[/]",
            )
            self.progress.reset(self.progress_bars[runner_id])

    def report_task_success(self, runner_id: int, task_id: int):
        task = self.tasks[task_id]
        if self.show_done:
            self.progress.log(f"[green]Task [yellow]{task.name}[/] completed 🎉")

        self.update_thread_progress(runner_id, completed=True)

    def report_task_error(self, runner_id: int, task_id: int):
        task = self.tasks[task_id]
        self.progress.log(f"[red]Error processing {task.name}[/red]")
        self.progress.log(traceback.format_exc())
        self.update_thread_progress(
            runner_id, description=f"[{runner_id}] Error processing [red]{task.name}", refresh=True
        )

    def report_task_end(self, runner_id: int, task: int):
        done = sum(1 for r in self.results.values() if not r.is_error)
        errors = sum(1 for r in self.results.values() if r.is_error)
        self.progress.update(
            self.progress_bars[-1],
            advance=1,
            description=f"Done: {done}, errors: {errors}, remaining: {len(self.tasks) - done - errors}",
        )

    def report_thread_end(self, runner_id: int):
        self.update_thread_progress(runner_id, completed=True)

    def report_all_end(self):
        self.progress.update(self.progress_bars[-1], completed=True)

    def report_thread_error(self, runner_id: int):
        self.progress.log(f"[red]Error in thread {runner_id}[/red]")
        self.progress.log(traceback.format_exc())
        if self.show_threads:
            self.update_thread_progress(
                runner_id,
                description=f"[{runner_id}] Error in thread, thread killed.",
                refresh=True,
                completed=True,
            )

    def update_thread_progress(self, runner_id: int, **kwargs):
        if self.show_threads:
            self.progress.update(self.progress_bars[runner_id], **kwargs)


if __name__ == "__main__":
    parallel = Parallel(n_jobs=4)

    def test_task(args):
        import random
        from time import sleep

        sleep(random.random())
        if random.random() < 0.1:
            raise ValueError("Random error")

        return args

    for i in range(30):
        parallel.add(test_task, title=f"Task {i}")(i)

    print(parallel.run())
