TRAINING THE VERIFIER

To train the reward model using trl library
## Create the reasoning dataset
#### Create the reasoning dataset for reward modelling from the modified questions and options

Run:

```./dataset/generate_reasoning.sh```

For training:
- Clone the _trl_ repository from _github_

```git clone https://github.com/huggingface/trl.git```

- Run the _trl/examples/scripts/reward_modeling.py_ script after adjusting the path and parameter variables
    The parameters used for training our llama-2-7B0chat model are mentioned in parameters.txt