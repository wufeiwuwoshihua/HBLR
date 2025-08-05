import json
import os
from tqdm import tqdm
from utils import OpenAIModel
import argparse
import re
import sys
from utils import LLM_response_self

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
                
    def load_in_context_examples_verifier(self):
        file_path = os.path.join('./verification', f'{self.dataset_name}', 'verification.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_dataset(self):
        file_path = os.path.join('./results', f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        with open(file_path, 'r') as f:
            translation = json.load(f)
        return translation
    
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_a(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        context = self.index_context(record['context'].strip())
        question = record['question'].strip()
        options = '\n'.join([opt.strip() for opt in record['options']])
        full_prompt = full_prompt.replace('[[PROBLEM]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        return full_prompt

    def construct_prompt_c(self, example, in_context_examples_solve):
        full_prompt = in_context_examples_solve
        context = example['context']
        execution = example['execution']
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[EXECUTION]]', execution)
        return full_prompt
    
    def construct_prompt_c_prontoqa(self, example, in_context_examples_solve):
        full_prompt = in_context_examples_solve
        context = example['context']
        execution = example['execution']
        origin = example['original_context']
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[EXECUTION]]', execution)
        full_prompt = full_prompt.replace('[[ORIGIN]]', origin)
        return full_prompt

    def post_process_a(self, response_a):
        response_a = str(response_a)
        context_start = response_a.find('"context":') + 10
        context_end = response_a.find('",\n"Question"')
        context = response_a[context_start:context_end].strip()
        question_start = response_a.find('"Question":') + 11
        question_end = response_a[question_start:].find('"}') + question_start
        question = response_a[question_start:question_end].strip()
        return context, question
    
    def post_process_c(self, response_c):
        try:
            pattern_bracket = r"Final answer: \{(.*?)\}"
            match = re.search(pattern_bracket, response_c)
            if match:
                return match.group(1)
            
            pattern_direct = r'\{(\w+)\}'
            match = re.search(pattern_direct, response_c, re.IGNORECASE)
            if match:
                return match.group(1).lower()
            return "No final answer found in the text."
        except:
            return "No final answer found in the text."

    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        else:
            final_answer = 'C'      
        return final_answer
    
    def reasoning_graph_generation(self):
        # load raw dataset
        in_context_examples_verifier = self.load_in_context_examples_verifier()
        
        dataset = self.load_dataset()
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        
        outputs = []
        error_output = []
        
        for example in tqdm(dataset):
            try:
                if self.dataset_name == 'ProntoQA':
                    prompts_c = self.construct_prompt_c_prontoqa(example, in_context_examples_verifier)
                else:
                    prompts_c = self.construct_prompt_c(example, in_context_examples_verifier)
                responses_c, finish_reason = LLM_response_self(prompts_c, self.model_name)

                final_answer = self.post_process_c(responses_c)
                final_choice = self.final_process(final_answer)

                output = {'id': example['id'],
                    'question': example['question'],
                    'context': example['context'],
                    'verification': responses_c,
                    'original_answer': example['predicted_choice'],
                    'predicted_answer': final_answer, 
                    'answer': example['answer'], 
                    'predicted_choice': final_choice,}
                outputs.append(output)
                
                with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_verified.json'), 'w') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error = {'id': example['id']}
                error_output.append(error)
                try:
                    with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_verified_error.json'), 'w') as f:
                            json.dump(error_output, f, indent=2, ensure_ascii=False)                 
                except:
                    print("Error in saving error output")
                continue            
            
        # save outputs        
        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_verified.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./verified_results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.reasoning_graph_generation()