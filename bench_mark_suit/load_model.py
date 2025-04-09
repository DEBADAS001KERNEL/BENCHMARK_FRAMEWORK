# it helps us to load the perticular model. here im using gemma -3-12b
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_gemma_model(model_name="google/gemma-3-12B"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device
