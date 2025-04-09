#Add multiple model like:
'''
Gemma-3B": "google/gemma-1.1-3b-it",
"LLaMA-2-7B": "meta-llama/Llama-2-7b-chat-hf",
rtc. but we can add more models or multyipole datas et if nedde.
'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt


def loadmodelandtokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def format_prompt(question, choices):
    prompt = question.strip() + "\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + i)}. {choice.strip()}\n"
    prompt += "\nAnswer:"
    return prompt

def get_answer(output_text):
    idx = output_text.find("Answer:")
    return output_text[idx+len("Answer:"):].strip()[0].upper() if idx != -1 else "?"

def evaluatemmlu(model_name, subject="college_mathematics", num_questions=10):
    tokenizer, model, device = loadmodelandtokenizer(model_name)
    dataset = load_dataset("cais/mmlu", subject, split="test").select(range(num_questions))

    correct = 0 # start with zero(0)
    for example in dataset:
        question = example["question"]
        choices = example["choices"]["text"]
        correct_answer = example["answer_key"]

        prompt = format_prompt(question, choices)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = get_answer(output_text)

        if predicted == correct_answer:
            correct += 1

    accuracy = (correct / num_questions) # accurecy checking logic
    return accuracy

def plot_results(results_dict):
    models = list(results_dict.keys())
    accuracies = [v * 100 for v in results_dict.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.title("MMLU Accuracy Comparison")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.ylim(0, 100)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Main benchmarking logic
if __name__ == "__main__":
    models_to_test = {
        "Gemma-3B": "google/gemma-1.1-3b-it",
        "LLaMA-2-7B": "meta-llama/Llama-2-7b-chat-hf",
        
    }

    results = {}
    for name, model_id in models_to_test.items():
        print(f"\n🔍 Evaluating {name}...")
        acc = evaluatemmlu(model_id)
        print(f" {name} Accuracy: {acc:.2%}")
        results[name] = acc

    plot_results(results)
