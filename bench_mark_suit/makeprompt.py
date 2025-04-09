#  formating our ptompt . making our prompt structuree.

def build_prompt(question, choices):
    prompt = question.strip() + "\n"
    for idx, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + idx)}. {choice.strip()}\n"
    prompt += "\nAnswer:"
    return prompt
