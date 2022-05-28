class FunctionCallCounter:
    """A class to count the number of calls to a function"""

    def __init__(self) -> None:
        """Initializes the counter"""
        self._count = 0
        self._uniques = set()

    def __call__(self, state: int, value: float) -> None:
        """Increase the counter
        :param state: The value for which the evaluator was called
        :param value: The value returned by the evaluator
        """
        self._count += 1
        self._uniques.add(state)

    @property
    def unique_calls(self) -> int:
        """Number of different values for which the evaluator was called for.
        This is the true post-perfect-caching count of the number of calls
        and therefore a good metric of the speed given evaluation is slow.
        :return: number of calls with different values to the evaluator.
        """
        return len(self._uniques)

    @property
    def total_calls(self) -> int:
        """Property containing the number of calls made to the evaluator.
        This can be artificially reduced by caching calls and therefore
        not the best metric of efficiency.
        :return: number of calls to the evaluator in total.
        """
        return self._count

    def reset(self) -> None:
        """Reset the counter back to 0"""
        self._count = 0
        self._uniques.clear()
