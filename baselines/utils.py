import requests
import json
import re

def remove_think_content(text):

    pattern = r"<think>.*?</think>"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return text[match.end():].lstrip()
    else:
        return text


from openai import OpenAI

def LLM_response_self(content, model):
    
    return chat_completion.choices[0].message.content