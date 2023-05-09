# @title Download Model
import pandas as pd
import re
import os
import datetime
import time
import random
from typing import Optional, Union, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# name_model = "pythainlp/wangchanglm-7.5B-sft-adapter-merged-sharded"
model = AutoModelForCausalLM.from_pretrained(
    name_model,
    return_dict=True,
    load_in_8bit=True,  # try use load_in_8bit=False for use fp16 :D
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="./",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-7.5B")
