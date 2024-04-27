from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import pandas as pd


CLASSIFICATION_INTRO = """
You are designed to categorize questions according to Costa's Levels of Questioning. Each level has a distinct focus:

- Level 1: Gathering and recalling information. Example: 'What is the command to list files in a directory?'
- Level 2: Processing and making sense of gathered information. Example: 'Hello? How does the 'ls' command organize the files it lists?'
- Level 3: Applying, analyzing, and evaluating information. Example: 'How would the process of file management change if the 'ls' command were significantly slower?'

Level 0 questions do not fit into the above categories and are typically rhetorical or lack a clear informational purpose. These are rare!

When classifying the questions, consider the depth and intent behind each question. You will be given a series of questions to classify, often with contextual sentences preceding them. Focus on classifying the last question provided, using the context for better understanding.

Respond with ONLY the level number (0, 1, 2, or 3) that best fits the question based on your understanding. Aim to use levels 1, 2, and 3 appropriately, reserving level 0 for questions that clearly do not seek an informational answer. Your judgment is valuable, and nuanced responses are encouraged.
"""


# CLASSIFICATION_INTRO = """You are designed to categorize questions according to Costa's Levels of Questioning. 
# Level 1 questions focus on gathering and recalling information. An example of a Level 1 question is "What is the command to list files in a directory?"
# Level 2 questions focus on making sense of gathered information. An example of a Level 2 question is "How does the 'ls' command work?"
# Level 3 questions focus on applying and evaluating information. An example of a Level 3 question is "What would the world be like if the 'ls' command didn't exist?"
# Level 0 questions are rhetorical questions or other sorts that don't fit the model. Be careful to classify these questions correctly, thinking about the intent behind the question.
# You will be given a series of questions to classify. Please provide the classification for each question. Try your best, don't worry about being perfect. 
# Make any assumptions you need to make and do not hesitate to use levels 2 and 3 if you think they are appropriate. 0 level questions are rare.
# If there are multiple questions, please answer the last question- everything before that is for context.
# You can respond with 0, 1, 2, or 3 to indicate the classification. Please respond with the number only.
# """


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

    data = data.head(10) # only use the first 10 questions

    # create the prompt
    prompt = []

    prompt.append({"role": "system", "content": CLASSIFICATION_INTRO})

    # add the questions and labels

    for _, row in data.iterrows():
        prompt.append({"role": "user", "content": row['question']})
        prompt.append({"role": "assistant", "content": row['label']})

    # append the user's question
    prompt.append({"role": "user", "content": f"All of the previous examples have been correct. Please use your best judgement in categorizing this question in context: {question}"})

    print(prompt)

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



def run_model(messages, temperature=0.6, top_p=0.9, max_new_tokens=1024):
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
            max_new_tokens=max_new_tokens,
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
    response = run_model(messages, temperature=0.2, top_p=0.9, max_new_tokens=1)
    print(response)
    return response
    # if response in ['0', '1', '2', '3']:
    #     return int(response)
    # else:
    #     return 0



def summarize_transcript(transcript: str) -> str:
    """
    Summarize the transcript using the model
    """

    # cut transcript to 8192 tokens
    transcript = transcript[:8192]

    PROMPT = """You are a helpful assistant for a classroom designed to summarize the teacher's lecture. 
    Do not include any prefix, just summarize the teacher's lecture. Keep the summary concise (2 paragraphs) and informative.
    For example, if the teacher was talking about the history of the USA, you could summarize by saying "This lecture was about the history of the USA, including the Revolutionary War and the Civil War"
    and then go into more detail about the Revolutionary War and the Civil War, as needed.
    """
    
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"{transcript}"},
    ]
    return run_model(messages)


if __name__ == "__main__":
    print(categorize_question("Hello? What is the capital of France?"))
    print(summarize_transcript("The capital of France is Paris."))







