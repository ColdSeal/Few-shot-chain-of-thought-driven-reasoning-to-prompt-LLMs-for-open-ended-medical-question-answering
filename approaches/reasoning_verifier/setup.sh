# conda env create -f rag.yml
git clone https://github.com/huggingface/trl.git
cp reward_modeling_llama70bchat.py ./trl/examples/scripts/
# conda activate rag
# cd trl/examples/scripts/
python trl/examples/scripts/reward_modeling_llama70bchat.py
