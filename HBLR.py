import json
import os
from tqdm import tqdm
from utils import *
import argparse
import re
import sys

def extract_final_output_new(text: str) -> str | None:
    pattern = r"### Final Reformatted Output(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_logic_program_block(text, type):
    if type == "FOLIO" or "ProofWriter" or "ProverQA":
        pattern = re.compile(r"(Premises:.*?Question:.*?$)", re.DOTALL)
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1)  # 取最后一个匹配结果
        else:
            return "None"
    elif type == "ProntoQA":
        pattern = re.compile(r"(Predicates:.*?Query:.*?$)", re.DOTALL)
        match = pattern.search(text)
        if match:
            return match.group(1)
        else:
            return "None"
    
    elif type == "LogicalDeduction":
        pattern = re.compile(r"(Domain:.*?Query:.*?$)", re.DOTALL)
        match = pattern.search(text)
        if match:
            return match.group(1)
        else:
            return "None"
        

        
def extract_predicates_and_logic_sections(text: str):
    if "Predicates:" not in text:
        raise ValueError("Missing 'Predicates:' section")

    _, rest = text.split("Predicates:", 1)

    if "Premises:" not in rest:
        raise ValueError("Missing 'Premises:' section")

    predicates_part, logic_part = rest.split("Premises:", 1)

    predicates = predicates_part.strip()
    logic = "Premises:" + logic_part.strip()

    return predicates, logic


def extract_final_output(text):
    premises = []
    conclusion = []
    lines = text.strip().splitlines()
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("Premises:"):
            current_section = "premises"
            continue
        elif line.startswith("Conclusion:"):
            current_section = "conclusion"
            continue

        if current_section == "premises" and line:
            premises.append(line)
        elif current_section == "conclusion" and line:
            conclusion.append(line)
    premises_str = "\n".join(premises)
    conclusion_str = "\n".join(conclusion)
    
    return premises_str, conclusion_str


class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
            
    def load_in_context_examples_trans(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'translation.txt')
        with open(file_path, encoding="utf-8") as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_select(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'select.txt')
        with open(file_path, encoding="utf-8") as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_examples_plan(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'backward.txt')
        with open(file_path, encoding="utf-8") as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_solve(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'solver.txt')
        with open(file_path, encoding="utf-8") as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json'), encoding="utf-8") as f:
            raw_dataset = json.load(f)
        return raw_dataset[:200]
    
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_a(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        context = record['context']
        question = record['question'].strip()
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        if self.dataset_name in ['AR-LSAT', 'LogicalDeduction']:
            choices = "\n".join(record['options'])
            full_prompt = full_prompt.replace('[[CHOICES]]', choices)
        return full_prompt

    def construct_prompt_b(self, responses_a, in_context_examples_plan):
        full_prompt = in_context_examples_plan
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_a)
        return full_prompt

    def construct_prompt_c(self, responses_a, responses_b, in_context_examples_solve):
        full_prompt = in_context_examples_solve
        plan = responses_b
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_a)
        full_prompt = full_prompt.replace('[[PLAN]]', plan)
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
        pattern_bracket = r"Final answer: \{([A-E])\}"
        match = re.search(pattern_bracket, response_c)
        if match:
            answers =  match.group(1)
            return answers
        pattern_direct = r'\{(\w+)\}'
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    
    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    
    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples_trans = self.load_in_context_examples_trans()
        in_context_examples_plan = self.load_in_context_examples_plan()
        in_context_examples_solve = self.load_in_context_examples_solve()
        in_context_examples_select = self.load_in_context_examples_select()
        
        outputs = []
        error_output = []

        for example in tqdm(raw_dataset):
            question = example['question']
            try:
                print("Filer & Translating...")
                prompts_a = self.construct_prompt_a(example, in_context_examples_trans)
                responses_a = LLM_response_self(prompts_a, self.model_name)

                print("Selecting...")
                i = 0
                while i < 3: 
                    prompts_select = self.construct_prompt_b(responses_a, in_context_examples_select)
                    print("Prompt Select: ", prompts_select)
                    
                    # responses_a, _ = self.openai_api.generate(prompts_a)
                    responses_selected_origin = LLM_response_self(prompts_select, self.model_name)
                    # responses_selected_origin = LLM_response(prompts_select, self.model_name)
                    if self.dataset_name in ["FOLIO", "ProofWriter", "ProntoQA"]:
                        responses_selected = extract_final_output_new(responses_selected_origin)
                    else:
                        responses_selected = responses_selected_origin
                    # responses_selected = predicates + "\n" + "Premises:" + "\n" + premises + "\n" + "Conclusion:" + "\n" + conclusion
                    # print("Responses Selected: ", responses_selected)
                    if responses_selected != "None":
                        break
                    else:
                        responses_selected = responses_selected_origin
                        i += 1
                        continue

                print("Reasoning...")
                prompts_b = self.construct_prompt_b(responses_selected, in_context_examples_plan)
                responses_b = LLM_response_self(prompts_b, self.model_name)
            
                
                prompts_c = self.construct_prompt_c(responses_selected, responses_b, in_context_examples_solve)
                responses_c = LLM_response_self(prompts_c, self.model_name)

                final_answer = self.post_process_c(responses_c)
                final_choice = self.final_process(final_answer)
                
                output = {'id': example['id'], 
                        'question': question, 
                        'answer': example['answer'], 
                        'predicted_answer': final_answer,
                        'predicted_choice': final_choice,
                        'context': responses_selected,
                        'plan': responses_b,
                        'execution': responses_c}
                print(output)
                outputs.append(output)
                
                with open(os.path.join(self.save_path, f'Select_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)                    
                
            except Exception as e:
                print('Error in generating example: ', example['id'])
                print(e)
                error = {'id': example['id']}
                error_output.append(error)
                try:
                    with open(os.path.join(self.save_path, f'Select_{self.dataset_name}_{self.split}_{self.model_name}_error.json'), 'w', encoding='utf-8') as f:
                            json.dump(error_output, f, indent=2, ensure_ascii=False)         
                except:
                    print("Error in saving error output")
                continue
                            
        # save outputs        
        with open(os.path.join(self.save_path, f'Select_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    
    def update_answer(self, sample, translation, plan, output):
        final_answer = self.post_process_c(output)
        final_choice = self.final_process(final_answer)
        dict_output = {'id': sample['id'],
                       'question': sample['question'],
                       'original_context': sample['context'],
                       'context': translation,
                       'plan': plan,
                       'execution': output,
                       'predicted_answer': final_answer, 
                       'answer': sample['answer'],
                       'predicted_choice': final_choice}
        return dict_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
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


