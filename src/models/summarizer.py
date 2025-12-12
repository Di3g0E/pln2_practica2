
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_name: str = 't5-small', device: str = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading summarization model: {model_name}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        # T5 specific prefix
        prefix = ""
        if 't5' in self.tokenizer.name_or_path:
            prefix = "summarize: "
            
        inputs = self.tokenizer(prefix + text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                max_length=max_length, 
                min_length=min_length, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
