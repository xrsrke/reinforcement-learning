import torch
import pytest

from octopus.policy.reward import calculate_discounted_return_each_timestep

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