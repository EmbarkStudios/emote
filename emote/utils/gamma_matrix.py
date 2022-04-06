import torch

# Taken from https://github.com/jackharmer/agency (MIT License)


# Construct a gamma matrix for optimised discount calculations.
# Using this in combination with the discount() function below
# provides up to 100x speedup over a non gamma matrix variant.
#
# Gamma Matrix Form  [roll_length+1, roll_length]:
#    [0.99^0, 0.0,    0.0   ]
#    [0.99^1, 0.99^0, 0.0   ]
#    [0.99^2, 0.99^1, 0.99^0]
#    [0.99^3, 0.99^2, 0.99^1]
#
#
# This allow the discount to be calculated as a dot product of the
# reward matrix and the gammaMatrix in one calculation across the whole
# batch.
#
# Reward Matrix:  [num_rolls, roll_length+1]
def make_gamma_matrix(gamma: float, roll_length: int):
    gamma = torch.tensor(gamma, dtype=torch.float32)
    gamma_matrix = torch.zeros((roll_length + 1, roll_length), dtype=torch.float32)
    gamma_vector = torch.zeros((roll_length + 1), dtype=torch.float32)
    for cc in range(roll_length + 1):
        gamma_vector[cc] = pow(gamma, cc)
    for cc in range(roll_length):
        gamma_matrix[cc : (roll_length + 1), cc] = gamma_vector[
            0 : roll_length + 1 - cc
        ]
    return gamma_matrix


# Calculate the discounted return using a gamma matrix, see above.
#
#  Reward Matrix         *     Gamma Matrix                 = Discount Matrix
#  [num_rolls, roll_length+1] [roll_length+1, roll_length]   [num_rolls, roll_length]
#
#  [ r0, r1, ..., v]           [0.99^0, 0.0   ]
#  [ r0, r1, ..., v]     *     [0.99^1, 0.99^0]
#  [ r0, r1, ..., v]           [0.99^2, 0.99^1]
def discount(rewards: torch.tensor, values: torch.tensor, gamma_matrix: torch.tensor):
    # [num_rolls, roll_length + 1]
    reward_matrix = torch.cat([rewards, values], dim=1)
    # [num_rolls, roll_length]
    discount_matrix = torch.matmul(reward_matrix, gamma_matrix)  # dot product
    # Discount vector: [num_rolls * roll_length]
    return torch.reshape(
        discount_matrix, (discount_matrix.shape[0] * discount_matrix.shape[1], 1)
    )


def split_rollouts(data: torch.tensor, rollout_len: int):
    return data.view([data.shape[0] // rollout_len, rollout_len, *data.shape[1:]])
