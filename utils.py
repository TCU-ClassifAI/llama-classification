from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import pandas as pd


CLASSIFICATION_INTRO = """You are designed to categorize questions according to Costa's Levels of Questioning. 
Level 1 questions focus on gathering and recalling information. An example of a Level 1 question is "What is the command to list files in a directory?"
Level 2 questions focus on making sense of gathered information. An example of a Level 2 question is "How does the 'ls' command work?"
Level 3 questions focus on applying and evaluating information. An example of a Level 3 question is "What would the world be like if the 'ls' command didn't exist?"
Level 0 questions are rhetorical questions or other sorts that don't fit the model. Be careful to classify these questions correctly, thinking about the intent behind the question.
You will be given a series of questions to classify. Please provide the classification for each question.
You can respond with 0, 1, 2, or 3 to indicate the classification.
"""


def load_training_data() -> pd.DataFrame:
    """
    Loads the training data for the question classification task.

    Returns:
        pd.DataFrame: A DataFrame containing 'question' and 'label' columns.
    """

    data = pd.read_csv('questions.csv')
    # print(data.head())

    # Add "Question: " to the beginning of each question
    data['question'] = 'Question: ' + data['question']

    # format labels as one-character strings (0, 1, 2)
    data['label'] = data['label'].astype(str)

    # Remove the .0 from the label
    data['label'] = data['label'].str.replace('.0', '')

    return data



def generate_categorization_prompt(question: str) -> list:
    """
    Generate the prompt for categorizing questions
    """

    # format: 
    # [
    # {"role": "system", "content": "..." },
    # {"role": "user", "content": "..."}
    # {"role": "assistant", "content": "..."}
    #]

    # load the training data
    data = load_training_data()

    # in a random order, have the user classify each question, and then provide the correct classification

    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # create the prompt
    prompt = []

    prompt.append({"role": "system", "content": CLASSIFICATION_INTRO})

    for _, row in data.iterrows():
        prompt.append({"role": "user", "content": row['question']})
        prompt.append({"role": "assistant", "content": row['label']})

    # append the user's question
    prompt.append({"role": "user", "content": question})

    # print(prompt)
    return prompt





model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]



input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# response = outputs[0][input_ids.shape[-1]:]
# print(tokenizer.decode(response, skip_special_tokens=True))


def run_model(messages, temperature=0.6, top_p=0.9):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)  # Move the input_ids to GPU

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,  # .module is used to access the original model within DataParallel
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )

    response = outputs.squeeze()[input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


# from utils.categorize_utils import generate_categorization_prompt
# print(run_model(generate_categorization_prompt("What is the capital of France?")))


def categorize_question(question: str) -> str:
    """
    Categorize the question using the model
    """

    messages = generate_categorization_prompt(question)
    return run_model(messages)



def summarize_transcript(transcript: str) -> str:
    """
    Summarize the transcript using the model
    """

    # cut transcript to 8192 tokens
    transcript = transcript[:8192]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant for a classroom designed to summarize the teacher's lecture. Do not include any prefix, just summarize the teacher's lecture. Keep the summary concise (2 paragraphs) and informative."},
        {"role": "user", "content": f"{transcript}"},
    ]
    return run_model(messages)


if __name__ == "__main__":
    print(categorize_question("What is the capital of France?"))
    print(summarize_transcript("The capital of France is Paris."))







