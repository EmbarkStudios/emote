import argparse
import numpy as np
import torch
from torch import nn

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory import MemoryLoader


def create_layer(res_block, n_blocks, in_channels, out_channels, stride=1):
    """
    Create a layer with specified type and number of residual blocks.
    Args:
        res_block: residual block type, BasicBlock for ResNet-18, 34 or
                  BottleNeck for ResNet-50, 101, 152
        n_blocks: number of residual blocks
        in_channels: number of input channels
        out_channels: number of output channels
        stride: stride used in the first 3x3 convolution of the first residual block
        of the layer and 1x1 convolution for skip connection in that block
    Returns:
        Convolutional layer
    """
    layer = []
    for i in range(n_blocks):
        if i == 0:
            # Down-sample the feature map using input stride for the first block of the layer.
            layer.append(res_block(in_channels, out_channels,
                                   stride=stride, is_first_block=True))
        else:
            # Keep the feature map size same for the rest three blocks of the layer.
            # by setting stride=1 and is_first_block=False.
            # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152,
            # ResBlock.expansion = 1 for ResNet-18, 34.
            layer.append(res_block(out_channels * res_block.expansion, out_channels))

    return nn.Sequential(*layer)


class ResNetFeatures(nn.Module):
    def __init__(
            self,
            res_block,
            n_blocks_list,
            out_channels_list,
            num_channels=3
    ):
        """
        Args:
            res_block: residual block type, BasicBlock for ResNet-18, 34 or
                      BottleNeck for ResNet-50, 101, 152
            n_blocks_list: number of residual blocks for each conv layer (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x - conv5_x
            num_channels: the number of channels of input image
        """
        super().__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels,
                                             out_channels=64, kernel_size=7,
                                             stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2, padding=1))

        # Create four convolutional layers
        in_channels = 64
        # For the first block of the second layer, do not down-sample and use stride=1.
        self.conv2_x = create_layer(res_block, n_blocks_list[0],
                                    in_channels, out_channels_list[0], stride=1)

        # For the first blocks of conv3_x - conv5_x layers, perform down-sampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152,
        # ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = create_layer(res_block, n_blocks_list[1],
                                    out_channels_list[0] * res_block.expansion,
                                    out_channels_list[1], stride=2)
        self.conv4_x = create_layer(res_block, n_blocks_list[2],
                                    out_channels_list[1] * res_block.expansion,
                                    out_channels_list[2], stride=2)
        self.conv5_x = create_layer(res_block, n_blocks_list[3],
                                    out_channels_list[2] * res_block.expansion,
                                    out_channels_list[3], stride=2)

        # Average pooling (used in classification head)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: input image
        Returns:
            c2: feature maps after conv2_x
            c3: feature maps after conv3_x
            c4: feature maps after conv4_x
            c5: feature maps after conv5_x
            y: output class
        """
        x = self.conv1(x)

        # Feature maps
        c2 = self.conv2_x(x)
        c3 = self.conv3_x(c2)
        c4 = self.conv4_x(c3)
        c5 = self.conv5_x(c4)

        # Classification head
        y = self.avg_pool(c5)
        y = y.reshape(y.shape[0], -1)

        return c2, c3, c4, c5, y


class BasicBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, is_first_block=False):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and
                    (b) 1x1 convolution used for down-sampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to
        # perform the add operations.
        self.down_sample = None
        if is_first_block and stride != 1:
            self.down_sample = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=1,
                                                       stride=stride,
                                                       padding=0),
                                             nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block output
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.down_sample:
            identity = self.down_sample(identity)
        x += identity
        x = self.relu(x)

        return x


def create_and_load_table(
        action_size: int,
        observation_size: int,
        use_terminal: bool,
        memory_max_size: int,
        memory_path: str,
        observation_key: str = "features"
):
    device = torch.device('cpu')
    spaces = MDPSpace(
        rewards=BoxSpace(dtype=np.float32, shape=(1,)),
        actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
        state=DictSpace(
            {
                observation_key: BoxSpace(
                    dtype=np.float32,
                    shape=tuple([observation_size, ])
                )
            }
        ),
    )
    table = DictObsNStepTable(
        spaces=spaces,
        use_terminal_column=use_terminal,
        maxlen=memory_max_size,
        device=device,
    )
    table.restore(memory_path)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--action-count", type=int, default=36)
    parser.add_argument("--observation-count", type=int, default=182)
    parser.add_argument("--observation-key", type=str, default="features")
    parser.add_argument("--vision-size", type=int, default=30)
    parser.add_argument("--memory-max-size", type=int, default=100000)
    parser.add_argument("--terminal-masking", action="store_true")

    arg = parser.parse_args()

    table = create_and_load_table(
        action_size=arg.action_count,
        observation_size=arg.observation_count + arg.vision_size * arg.vision_size,
        use_terminal=arg.terminal_masking,
        memory_path=arg.path_to_buffer,
        memory_max_size=arg.memory_max_size,
        observation_key=arg.observation_key
    )

    resnet_model = ResNetFeatures(
        res_block=BasicBlock,
        n_blocks_list=[2, 2, 2, 2],
        num_channels=1,
        out_channels_list=[64, 128, 256, 512]
    )

    batch_size = 10
    rollout_length = 1

    data_loader = MemoryLoader(
        table,
        batch_size // rollout_length,
        rollout_length,
        "batch_size",
        data_group="rl_loader",
    )

    data_iter = iter(data_loader)

    batch = next(data_iter)

    vision_input = batch["rl_loader"]['observation'][arg.observation_key][:, -arg.vision_size * arg.vision_size:]

    c2, c3, c4, c5, y = resnet_model.forward(vision_input.view(batch_size, 1, arg.vision_size, arg.vision_size))
    print(vision_input.shape)
    print(c2.shape, c3.shape, c4.shape, c5.shape, y.shape)
