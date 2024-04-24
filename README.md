# Categorization and Summarization of Transcript API

This is a simple API that categorizes and summarizes a given text. The API is built using Flask. Both the categorization and summarization models done using the Llama-3 models. 

The model is [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Requirements
Install the required packages using the following command:
`pip install -r requirements.txt`

At the moment, this is run on a 4090 GPU. It can be run on a weaker GPU, though you may have to use less precision for the model.

Get the model from the Hugging Face model hub [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). You will have to give meta your email to get access to the model.

Be sure to include your [huggingface API key](https://huggingface.co/docs/hub/en/security-tokens)

## Running the API
To run the API, use the following command:
`python app.py`

The API will be running on `http://localhost:5002` by default.

## Sample request for categorization
`curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the square root of 5?"}' http://localhost:5002/categorize`

Note: This does better if you include more context (previous sentences) in the question. For example, "Okay, we have learned about the Pythagorean theorem. What is the square root of 5?"

### Sample response for categorization
`{"response":1}` (signifying this is a level 1 question, according to [Costa's levels of questioning](https://www.fortbendisd.com/cms/lib/TX01917858/Centricity/Domain/2615/Costas_3_Levels_of_Thinking.pdf))

## Sample request for summarization
`curl localhost:5002/summarize -X POST -H "Content-Type: application/json" -d '{"text":"The quick brown fox jumps over the lazy dog."}'`
### Sample response for summarization
`{"response":"The quick brown fox jumps over the lazy dog. This is 'a classroom transcript,' where I am to sum up what the teachers are teaching."}`

## Healthcheck
`curl localhost:5002/healthcheck`

### Return
`{"status": "ok"}`


### Tips for prompting

- Use more context in the question for better categorization.
- To remove the 'chattiness' of the model, specify "respond only as shown, with no explanatory text" in the system prompt. 
- ALWAYS provice an example, ie, one-shot in-context learning.
