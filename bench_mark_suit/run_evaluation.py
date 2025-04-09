# run_ the whole evaluation on samples.
from load_model import load_gemma_model
from load_mmlu_dataset import load_mmlu
from evaluate_single import evaluate_example

def main():
    tokenizer, model, device = load_gemma_model()
    dataset = load_mmlu(subject="college_mathematics")  
    results = []

    for i, example in enumerate(dataset.select(range(10))):  # TEST ON 10 Smaple
        pred, prompt, full = evaluate_example(example, tokenizer, model, device)
        correct = pred == example["answer_key"]
        results.append(correct)

        print(f"\nPrompt:\n{prompt}")
        print(f"Model Output:\n{full}")
        print(f"Predicted: {pred} | Correct: {example['answer_key']}")

    accuracy = sum(results) / len(results)
    print(f"\nAccuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
