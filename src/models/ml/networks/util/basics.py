from typing import Any, Iterator, List, Optional, Tuple, Union

import torch


def exists(x: Any) -> bool:
    return x is not None


def noop(*args: Any, **kwargs: Any) -> None:
    pass


def is_odd(n: int) -> bool:
    return (n % 2) == 1


def default(val: Optional[Any], d: Any) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl: Any) -> Iterator[Any]:
    while True:
        for data in dl:
            yield data


def num_to_groups(num: int, divisor: int) -> List[int]:
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(
    shape: Union[Tuple[int, ...], List[int]], prob: float, device: torch.device
) -> torch.Tensor:
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x: Any) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    return all([isinstance(el, str) for el in x])


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple):
    assert a.ndim == t.ndim == 1
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
