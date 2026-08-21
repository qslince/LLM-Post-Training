# Notebooks

Step-by-step notebooks that walk through each stage of the post-training pipeline. Each notebook contains the full code with saved outputs, so you can follow the logic and see the results without re-running anything.

| Notebook | Description |
|----------|-------------|
| [SFT.ipynb](SFT.ipynb) | Supervised Fine-Tuning on Qwen2.5-0.5B with OpenHermes 2.5 |
| [Reward_Model.ipynb](Reward_Model.ipynb) | Bradley-Terry reward model training on UltraFeedback |
| [Pair_Generation.ipynb](Pair_Generation.ipynb) | Generating preference pairs: 8 responses per prompt, RM scoring, best/worst selection |
| [DPO.ipynb](DPO.ipynb) | Direct Preference Optimization with LoRA, merge & generation |
| [GRPO.ipynb](GRPO.ipynb) | Group Relative Policy Optimization with LoRA and RM as reward function |
