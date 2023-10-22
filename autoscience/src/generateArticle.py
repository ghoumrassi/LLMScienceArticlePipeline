import os
import re
import openai

from dotenv import load_dotenv

load_dotenv("autoscience/.env")

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


def generateSummary(variables_dict, max_tokens):
        with open(r'autoscience\prompts\summary_gen_1.txt', 'r') as file:
            prompt = file.read()

        # Find all variables in the prompt
        variables = re.findall(r'\[(.+?)\]', prompt)

        # Replace the variables with their corresponding values in Python
        for var in variables:
            if var in variables_dict:
                value = str(variables_dict[var])
                prompt = prompt.replace('%[' + var + ']', value)
        
        # Generate text from the prompt using OpenAI completions API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            max_tokens=max_tokens, 
            n=1, 
            stop=None, 
            temperature=0.7,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a world-renowned science and technology journalist."
                },
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response['choices'][0]['message']['content']

        return generated_text


if __name__ == "__main__":
    refs = []
    for i in range(1, 4):
        with open(f"autoscience/data/raw/gptbar_{i}.txt", 'r', encoding='utf8') as f:
            refs.append(f.read())
    v = {
        "target audience": "young professionals aged 20-30",
        "desired tone": "optimistic about the future of scientific progress",
        "Reference text 1": refs[0],
        "Reference text 2": refs[1],
        "Reference text 3": refs[2],
        "list of reference numbers that shouldn't be quoted directly": [1, 2],
        "list of reference numbers that can be quoted": [3],

    }
    summary = generateSummary(v, 4000)

    print(summary)