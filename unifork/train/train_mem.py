import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)


from unifork.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
