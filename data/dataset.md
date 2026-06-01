The task-completion datasets relevant to GUI agents are **AndroidControl** [1], **AITZ** [2], and **GUIOdyssey** [3]. Please download the original data for each dataset and preprocess them into a unified JSON list format. Each sample should follow the format below and be compatible with the evaluation script provided in this repository:

```json
{
  "task": "Open the Cx File Explorer and rename the Flowers folder to Flora.",
  "image_path": "/data3/datasets/android_control/images/episode_27_screenshot_1.png",
  "action": "CLICK <point>[[950, 255]]</point>"
}
```

The `action` field should follow the action space defined in **OS-Atlas** [4].

For the GUI knowledge benchmarks, namely **MMBench-GUI L1** [5] and **GUI Knowledge Bench** [6], you only need to download the original datasets and run the evaluation scripts provided in this repository directly. No additional data preprocessing is required.

### References

[1] *On the Effects of Data Scale on UI Control Agents*. [https://arxiv.org/abs/2406.03679](https://arxiv.org/abs/2406.03679)

[2] *Android in the Zoo: Chain-of-Action-Thought for GUI Agents*. [https://arxiv.org/abs/2403.02713](https://arxiv.org/abs/2403.02713)

[3] *GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices*. [https://arxiv.org/abs/2406.08451](https://arxiv.org/abs/2406.08451)

[4] *OS-ATLAS: A Foundation Action Model for Generalist GUI Agents*. [https://arxiv.org/abs/2410.23218](https://arxiv.org/abs/2410.23218)

[5] *MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agents*. [https://arxiv.org/abs/2507.19478](https://arxiv.org/abs/2507.19478)

[6] *GUI Knowledge Bench: Revealing the Knowledge Gap of VLMs in GUI Tasks*. [https://arxiv.org/abs/2510.26098](https://arxiv.org/abs/2510.26098)
