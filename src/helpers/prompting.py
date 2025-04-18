import random

random.seed(42)

def select_random_system_prompt(system_prompts):
    """Randomly selects a system prompt from the list and returns its ID and text."""
    selected_prompt = random.choice(system_prompts["system_prompts"])  # Seed set for reproducibility
    return selected_prompt["id"], selected_prompt["prompt"]

def select_random_user_prompt(user_prompts, metric_name, metric_value, user_prompt_id):
    """Randomly selects a user prompt from the list and returns its ID and text."""
    selected_prompts = user_prompts.get(metric_name)
    if selected_prompts:
        selected_prompt = random.choice(selected_prompts[user_prompt_id])
        return selected_prompt["id"], selected_prompt["prompt"].replace("{VALUE}", str(metric_value))
    return None, None

def select_control_token_explanation(control_tokens, metric_name, metric_value):
    """Selects a control token explanation based on the metric name and value."""
    control_token = control_tokens.get(metric_name)
    if control_token:
        return control_token["explanation"].replace("{VALUE}", str(metric_value))
    return None

def select_random_control_token_examples(control_tokens, metric_name, num_examples=3):
    """Selects a random set of control token examples based on the metric name and value."""
    control_token = control_tokens.get(metric_name)
    if control_token and "examples" in control_token:
        # examples = random.sample(control_token["examples"], num_examples)
        examples = random.sample(control_token["examples"], min(num_examples, len(control_token["examples"])))
        return examples
    return []

def create_user_prompt(user_prompts, metric_name, metric_value, user_prompt_id, text, explanation=None, examples=None):
    """Constructs a user prompt based on the user prompt type."""
    # Access the list of prompts for the selected metric (ARI, FKGL, etc.)
    metric_prompts = user_prompts.get(metric_name)

    if metric_prompts:
        prompt_index = {
            "no_token": 0,
            "token": 1,
            "token_explanation": 2,
            "token_explanation_examples": 3
        }.get(user_prompt_id, 0)  # Default to 0 if not found
        
        prompt_id = metric_prompts[prompt_index]["id"]
        prompt_text = metric_prompts[prompt_index]["prompt"]

        if "explanation" in user_prompt_id and explanation:
            prompt_text = prompt_text.replace("{EXPLANATION}", explanation)
        
        if "examples" in user_prompt_id and examples:
            for i, example in enumerate(examples, start=1):
                prompt_text = prompt_text.replace(f"{{EXAMPLE_{i}}}", example)

        prompt_text = prompt_text.replace("{VALUE}", str(metric_value))
        prompt_text = prompt_text.replace("{TEXT}", text)
        prompt_text = prompt_text.replace("{EOT_TOKEN}", "<|eot_id|>")
        # print(f"Prompt before return: {prompt_text}") # for debugging
        return prompt_id, prompt_text
    
    # Fallback: If the metric is not found, return a default prompt
    return user_prompt_id, f"Simplify the following text: \n\n{text}"

def format_prompt_with_special_tokens(system_prompt, user_prompt):
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> {system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|> {user_prompt}\n<|start_header_id|>assistant<|end_header_id|> """
    return formatted_prompt

def format_completion_with_special_tokens(completion):
    return f"{completion} <|eot_id|>"

def create_inference_prompt(text, metric_value, metric_name):
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> "
        f"<{metric_name}={metric_value}> {text} <|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|> "
    )