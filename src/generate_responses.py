import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env", override=True)
from Metrics import Metrics


OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_TOKEN)


def generate_response(messages, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def simplify(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Simplify this text: {input_text}"}
    ]
    response = generate_response(messages)

    return response

def simplify_with_reasoning(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Briefly describe the needs of the target group, if such is specified. Using the instructions you were given, briefly explain how you would simplify the following source text and why: {input_text}. Think step by step, and very explain the changes you would make bake to satisfy the instructions, including any calculations, if required. Do not yet generate the simplification! Be very brief."}
    ]
    reasoning = generate_response(messages)

    messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "user", "content": f"Now that you have developed a simplification strategy, generate the simplification. Only write the simplification. No other comments are allowed."})
    
    simplification = generate_response(messages)

    return reasoning, simplification

def simplify_with_consecutive_transformations(input_text, prompt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"You will perform a step-by-step simplification of the following text: {input_text}."},
        {"role": "user", "content": f"Start by performing a syntactic simplification. Reduce sentences to minimal clauses. You can split sentences into several, if needed."}
    ]
    syntactic_symplification = generate_response(messages)

    messages.append({"role": "assistant", "content": syntactic_symplification})
    messages.append({"role": "user", "content": f"Proceed with a lexical simplification. Substitute domain-specific and difficult words with simple words, if possible."})

    lexical_simplification = generate_response(messages)

    messages.append({"role": "assistant", "content": lexical_simplification})
    messages.append({"role": "user", "content": f"Now, feel free to use paraphrase or explanation for terms and concepts you think require it."})

    paraphrase_or_explanation = generate_response(messages)

    messages.append({"role": "assistant", "content": paraphrase_or_explanation})
    messages.append({"role": "user", "content": f"Now, generate the final simplification based on your previous thoughts. Output only the simplification. No notes or comments are allowed."})

    simplification = generate_response(messages)

    return syntactic_symplification, lexical_simplification, paraphrase_or_explanation, simplification

def simplify_with_corrections(input_text, prompt):

    initial_simplification = simplify(input_text, prompt)

    messages = [
        {"role": "user", "content": f"A text simplification assistant was given the following task formulation: {prompt}."},
        {"role": "user", "content": f"This is the source text: {input_text}."},
        {"role": "user", "content": f"Given the task formulation and the source text, critically assess the simplification. Be brief (1-2 sentences)."}
    ]

    judgement = generate_response(messages)

    messages.append({"role": "assistant", "content": judgement})
    messages.append({"role": "user", "content": f"Take into account your critical assessment of the simplification. Now it is your turn to further improve upon the simplification to align it with the task, but make only necessary changes. Output only the simplification, no notes or comments are allowed."})

    final_simplification = generate_response(messages)

    return initial_simplification, judgement, final_simplification

def main():

    experiment_name = "metacorrections" # reasoning, consecutive, basic, metacorrections
    dataset_name = "sample_sentence_imaging"
    # TODO setup iteration over our datasets
    input_text = "Neuroscientists have long utilized sophisticated imaging techniques, such as functional magnetic resonance imaging (fMRI), to investigate the intricate neural mechanisms underlying cognitive processes like memory formation, decision-making, and language comprehension."
    print("Input text:\n", input_text)

    # create output dir
    output_dir = f"./output/prompting/{experiment_name}/{MODEL}/{dataset_name}"
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open("./data/prompts/prompts.json", "r") as file:
        prompts = json.load(file)
        with open("./data/prompts/justifications.json", "r") as file:
            justifications = json.load(file)

    # Iterate through prompt categories and variations
    for category, variations in prompts.items():
        for prompt_id, prompt in variations.items():

            if experiment_name == "reasoning":
                reasoning, simplified_text = simplify_with_reasoning(input_text, prompt)
                output_data = {
                    "reasoning": reasoning
                }
            elif experiment_name == "consecutive":
                syntactic, lexical, paraphrase, simplified_text = simplify_with_consecutive_transformations(input_text, prompt)
                output_data = {
                    "syntactic_simplification": syntactic,
                    "lexical_simplification": lexical,
                    "paraphrase_or_explanation": paraphrase
                }
            elif experiment_name == "metacorrections":
                initial_simplification, judgement, simplified_text = simplify_with_corrections(input_text, prompt)
                initial_simplification_metrics = Metrics(initial_simplification).compute_metrics()
                output_data = {
                    "initial_simplification": initial_simplification,
                    "initial_simplification_metrics": initial_simplification_metrics,
                    "metajudgement": judgement
                }

            else:
                simplified_text = simplify(input_text, prompt)
                output_data = {}

            source_metrics = Metrics(input_text).compute_metrics()
            target_metrics = Metrics(simplified_text).compute_metrics()
            output_data.update({
                "category": category,
                "prompt_used": prompt,
                "prompt_id": prompt_id,
                "source_text": input_text,
                "target_text": simplified_text,
                "source_metrics": source_metrics,
                "target_metrics": target_metrics
            })

            output_file = f"{output_dir}/{category}_{prompt_id}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as file:
                json.dump(output_data, file, indent=4)

            print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()