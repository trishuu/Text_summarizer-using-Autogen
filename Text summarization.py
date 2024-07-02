# text_summarization.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import autogen

class TextSummarizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def summarize(self, text):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

class SummarizationAgent(autogen.Agent):
    def __init__(self, model_name):
        self.summarizer = TextSummarizer(model_name)
    
    def act(self, text):
        return self.summarizer.summarize(text)

agent1 = SummarizationAgent('facebook/bart-large-cnn')
agent2 = SummarizationAgent('t5-base')
autogen.add_agents([agent1, agent2])

text = "Hello Welcome to my short project I am Trisha who love to work with ML models, and creating my own Algorithms. My strength is to put 110% in whatever task I take up. This work I had done in only 3-4 hours if I was allowed to spare some more time I would have definitely give my top best. Thank you :) "
summaries = autogen.run_agents(text)
for i, summary in enumerate(summaries):
    print(f"Summary by agent {i+1}: {summary}")
