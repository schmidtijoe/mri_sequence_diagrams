import numpy as np
import typing


def array1d_to_value_find_nearest_idx(array: np.array, value: typing.Union[int, float]) -> int:
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def get_start_end_idx(array: np.ndarray, value: typing.Union[float, int], duration_in_ms: int) -> (int, int):
    start_idx = array1d_to_value_find_nearest_idx(array, value - int(duration_in_ms / 2))
    end_idx = array1d_to_value_find_nearest_idx(array, value + int(duration_in_ms / 2))
    return start_idx, end_idx
