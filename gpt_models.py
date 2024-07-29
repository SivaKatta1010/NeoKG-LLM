import time
import pandas as pd
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict, Any
from rankings.learning_to_rank import generate_context
from models.constants import GPT_API_KEY
from utils.metrics import calculate_all_scores

# Initialize the model and LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=GPT_API_KEY)

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

memory = ExtendedConversationBufferWindowMemory(k=0, ai_prefix="Physician", human_prefix="Patient", extra_variables=["context"])

template = """
Answer the question in conjunction with the following content.

Context:
{context}
Patient: {input}
Physician:
"""

PROMPT_TEMPLATE = PromptTemplate(input_variables=["context", "input"], template=template)

conversation = ConversationChain(llm=llm, prompt=PROMPT_TEMPLATE, verbose=True)

questions_df = pd.read_csv("/home/skatta14/HealthcareLLM/KGQA-270F/umlsclass/ExpertQA_Bio.csv", usecols=['Question', 'Answer'])
questions = questions_df['Question'].head(2).tolist()
reference_answers = questions_df['Answer'].head(2).tolist()

total_time, total_time1, total_time2 = 0, 0, 0
generated_answers = []

for question in tqdm(questions, desc="Processing Questions"):
    start_time = time.time()
    
    start_time1 = time.time()
    context = generate_context(question)
    total_time1 += time.time() - start_time1

    start_time2 = time.time()
    full_answer = conversation.predict(context=context, input=question)
    total_time2 += time.time() - start_time2

    answer_start = full_answer.find('Answer: [/INST]') + len('Answer: [/INST]')
    generated_answers.append(full_answer[answer_start:].strip())

    total_time += time.time() - start_time

all_scores = calculate_all_scores(reference_answers, generated_answers)

print(f"Average inference time: {total_time / len(questions):.2f} seconds")
print(f"Average Database API call time: {total_time1 / len(questions):.2f} seconds")
print(f"Average LLM inference time: {total_time2 / len(questions):.2f} seconds")
print(f"BLEU Score: {all_scores['BLEU']:.2f}")
print(f"Average ROUGE Scores: {all_scores['ROUGE']}")
print(f"BERTScore (BioBERT) - Precision: {all_scores['BERTScore_BioBERT'][0]:.2f}, Recall: {all_scores['BERTScore_BioBERT'][1]:.2f}, F1: {all_scores['BERTScore_BioBERT'][2]:.2f}")
print(f"BERTScore (PubMedBERT) - Precision: {all_scores['BERTScore_PubMedBERT'][0]:.2f}, Recall: {all_scores['BERTScore_PubMedBERT'][1]:.2f}, F1: {all_scores['BERTScore_PubMedBERT'][2]:.2f}")