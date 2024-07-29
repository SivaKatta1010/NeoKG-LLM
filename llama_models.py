import torch
import transformers
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from torch import cuda
from utils.metrics import calculate_all_scores
import pandas as pd
from tqdm import tqdm
import time
from rankings.learning_to_rank import generate_context

# Model and tokenizer setup
checkpoint = "meta-llama/llama-2-7b-chat-hf"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    quantization_config=nf4_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.eval()

# Define stopping criteria
stop_list = ["\nHuman:", "\n```\n"]
stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if input_ids.shape[1] >= stop_ids.shape[0] and torch.eq(input_ids[0, -stop_ids.shape[0]:], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# Initialize pipeline
pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
    max_new_tokens=2048,
    return_full_text=True,
    stopping_criteria=stopping_criteria,
    repetition_penalty=1.1,
    do_sample=True,
)

llama = HuggingFacePipeline(pipeline=pipeline)

class ExtendedConversationBufferWindowMemory(ConversationBufferWindowMemory):
    extra_variables: List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        return self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        d = super().load_memory_variables(inputs)
        d.pop("history", None)
        d.update({k: inputs.get(k) for k in self.extra_variables})
        return d

memory = ExtendedConversationBufferWindowMemory(k=0,
                                                ai_prefix="Physician",
                                                human_prefix="Patient",
                                                extra_variables=["context"])

template = """
<s>[INST] <<SYS>>
Answer the question in conjunction with the following content.
<</SYS>>

Context:
{context}

Question: {input}
Answer: [/INST]
"""

PROMPT = PromptTemplate(
    input_variables=["context", "input"], template=template
)

conversation = ConversationChain(
    llm=llama,
    memory=memory,
    prompt=PROMPT,
    verbose=True,
)

# Load dataset
questions_df = pd.read_csv("/home/skatta14/HealthcareLLM/KGQA-270F/umlsclass/ExpertQA_Bio.csv", usecols=['Question', 'Answer'])
questions = questions_df['Question'].tolist()
reference_answers = questions_df['Answer'].tolist()

# Process each question
total_time = 0
total_time1 = 0
total_time2 = 0
num_questions = len(questions)
generated_answers = []

for question in tqdm(questions, desc="Processing Questions"):
    start_time = time.time()
    
    start_time1 = time.time()
    context = generate_context(question)
    end_time1 = time.time()
    total_time1 += end_time1 - start_time1

    start_time2 = time.time()
    full_answer = conversation.predict(context=context, input=question)
    end_time2 = time.time()
    total_time2 += end_time2 - start_time2

    answer_start = full_answer.find('Answer: [/INST]') + len('Answer: [/INST]')
    answer = full_answer[answer_start:].strip()
    end_time = time.time()
    total_time += end_time - start_time
    
    generated_answers.append(answer)

all_scores = calculate_all_scores(reference_answers, generated_answers)

print(f"Average inference time: {total_time / len(questions):.2f} seconds")
print(f"Average Database API call time: {total_time1 / len(questions):.2f} seconds")
print(f"Average LLM inference time: {total_time2 / len(questions):.2f} seconds")
print(f"BLEU Score: {all_scores['BLEU']:.2f}")
print(f"Average ROUGE Scores: {all_scores['ROUGE']}")
print(f"BERTScore (BioBERT) - Precision: {all_scores['BERTScore_BioBERT'][0]:.2f}, Recall: {all_scores['BERTScore_BioBERT'][1]:.2f}, F1: {all_scores['BERTScore_BioBERT'][2]:.2f}")
print(f"BERTScore (PubMedBERT) - Precision: {all_scores['BERTScore_PubMedBERT'][0]:.2f}, Recall: {all_scores['BERTScore_PubMedBERT'][1]:.2f}, F1: {all_scores['BERTScore_PubMedBERT'][2]:.2f}")
