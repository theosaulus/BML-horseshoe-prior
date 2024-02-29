from collections import defaultdict
import time
from typing import Any, Callable, Dict, Union


def timeit(func: Callable[..., Any]) -> Callable[..., Union[Any, float]]:
    """A wrapper function to return the result of the function and the time it took to execute it."""

    def time_measured_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    return time_measured_func


class RuntimeMeter:
    """A context manager class to measure the time of various stages of the code take.

    with RuntimeMeter("train") as rm:
        # Training code
        #  ...
    with RuntimeMeter("eval") as rm:
        # Evaluation code
        #  ...
    with RuntimeMeter("train") as rm:   # second training phase, but the RuntimeMeter will add the time to the first training phase
        # Training code
        #  ...

    training_time = RuntimeMeter.get_runtime("train")
    eval_time = RuntimeMeter.get_runtime("eval")
    """

    stage_name_to_runtime: Dict[str, float] = defaultdict(lambda: 0)

    @staticmethod
    def get_stage_runtime(stage_name: str) -> float:
        """Return the cumulative time taken by the stage.
        If the stage_name is "total", it will return the total time taken by all stages.
        If the stage_name is not found, it will return 0.

        Args:
            stage_name (str): the name of the stage, as it was used in the context manager.

        Returns:
            float: the cumulative time taken by the stage.
        """
        if stage_name == "total":
            return sum(RuntimeMeter.stage_name_to_runtime.values())
        elif stage_name not in RuntimeMeter.stage_name_to_runtime:
            return 0
        else:
            return RuntimeMeter.stage_name_to_runtime[stage_name]

    @staticmethod
    def get_stage_runtimes() -> Dict[str, float]:
        """Return a dictionnary mapping the stage names to the cumulative time taken by the stage.

        Returns:
            Dict[str, float]: the dictionnary mapping the stage names to the cumulative time taken by the stage.
        """
        return dict(RuntimeMeter.stage_name_to_runtime)

    @staticmethod
    def get_total_runtime() -> float:
        """Return the total time taken by all stages.

        Returns:
            float: the total time taken by all stages.
        """
        return sum(RuntimeMeter.stage_name_to_runtime.values())

    def __init__(self, stage_name: str):
        self.stage_name = stage_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stage_name_to_runtime[self.stage_name] += time.time() - self.start_time


if __name__ == "__main__":
    import time
    import random

    def foo():
        time.sleep(0.1)

    def bar():
        time.sleep(0.2)

    for _ in range(3):
        with RuntimeMeter("foo"):
            foo()
        with RuntimeMeter("bar"):
            bar()

    print(RuntimeMeter.get_stage_runtime("foo"))
    print(RuntimeMeter.get_stage_runtime("bar"))
    print(RuntimeMeter.get_stage_runtime("total"))
