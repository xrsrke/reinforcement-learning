import torch
# from torch.testing import assertEqual
import pytest

from octopus.policy.reward import (
    calculate_discounted_return_each_timestep,
    calculate_advantages
)

@pytest.mark.parametrize(
    "rewards, discount_factor, expected_output",
    [
        (
            torch.tensor([5, 2, 3, 4]),
            0.99,
            torch.tensor([13.8015,  8.8904,  6.9600,  4.0000])
        ),
        (
            [torch.tensor(5), torch.tensor(2), torch.tensor(3), torch.tensor(4)],
            0.99,
            torch.tensor([13.8015,  8.8904,  6.9600,  4.0000])
        )
    ]
)
def test_calculate_discounted_return_each_timestep(rewards, discount_factor, expected_output):
    output = calculate_discounted_return_each_timestep(rewards, discount_factor)
    torch.testing.assert_close(output, expected_output)


def test_calculate_advantages():
    discounted_returns = torch.tensor([10, 20, 30, 40])
    q_values = torch.tensor([1, 2, 3, 4])
    expected_advantages = torch.tensor([9, 18, 27, 36])

    advantages = calculate_advantages(discounted_returns, q_values)

    assert (advantages == expected_advantages).all()

def test_calculate_advantages_raise_non_equal_length():
    discounted_returns = torch.tensor([10, 20, 30, 40])
    q_values = torch.tensor([1, 2])

    with pytest.raises(AssertionError):
        calculate_advantages(discounted_returns, q_values)