# Reinforcement-Learning-TD3-
Implement TD3 algorithm to train a car to move across two destinations in a map

### Params
- state_dim - a crop of 100x100x3 of the mask where car is present. This will act as the visual input
- action_dim - rotation angle for the car. Range [-5,5]
- max_action - 5


### Actor model 
```
Actor(
  (conv1): conv_block(
    (convblock): Sequential(
      (0): Conv2d(3, 12, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (conv2): conv_block(
    (convblock): Sequential(
      (0): Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (conv3): conv_block(
    (convblock): Sequential(
      (0): Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (maxpool_2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (reduce): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1))
  (gap): AdaptiveAvgPool2d(output_size=(1, 1))
  (last): Conv2d(10, 1, kernel_size=(1, 1), stride=(1, 1))
)

```
### Critic model
```
Critic(
  (conv1_1): conv_block(
    (convblock): Sequential(
      (0): Conv2d(3, 12, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (conv2_1): conv_block(
    (convblock): Sequential(
      (0): Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (conv3_1): conv_block(
    (convblock): Sequential(
      (0): Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (maxpool_1_1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (reduce_1_1): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1))
  (gap_1): AdaptiveAvgPool2d(output_size=(1, 1))
  (last_1_1): Conv2d(10, 1, kernel_size=(1, 1), stride=(1, 1))
  (last_1_2): Linear(in_features=2, out_features=10, bias=True)
  (last_1_3): Linear(in_features=10, out_features=1, bias=True)
  (conv1_2): conv_block(
    (convblock): Sequential(
      (0): Conv2d(3, 12, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (conv2_2): conv_block(
    (convblock): Sequential(
      (0): Conv2d(12, 24, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (conv3_2): conv_block(
    (convblock): Sequential(
      (0): Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.1)
    )
  )
  (maxpool_2_1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (reduce_2_1): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1))
  (gap_2): AdaptiveAvgPool2d(output_size=(1, 1))
  (last_2_1): Conv2d(10, 1, kernel_size=(1, 1), stride=(1, 1))
  (last_2_2): Linear(in_features=2, out_features=10, bias=True)
  (last_2_3): Linear(in_features=10, out_features=1, bias=True)
)

```
