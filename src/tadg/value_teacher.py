"""Read the pretrained V from the shared critic features without double-scaling goals."""


def value_from_features(critic, features, goal, state_dim, xi_dim):
    xi_start = state_dim + goal.shape[-1]
    root_start = xi_start + xi_dim
    if features.shape[-1] < root_start + 4:
        raise ValueError("Critic features do not contain state, goal, context and root error")
    # Critic.forward applies its own goal scale. Pass raw goal, not the scaled
    # goal block embedded in Q features. Keep input gradients for BPTT callers.
    return critic(features[..., :state_dim], goal,
                  features[..., xi_start:root_start], features[..., root_start:root_start + 4])
