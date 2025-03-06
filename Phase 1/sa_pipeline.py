import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
from typing import List

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            return ""
        return text

class SentimentAnalyzer:
    def __init__(self, model_name: str = "siebert/sentiment-roberta-large-english", 
                 batch_size: int = 256, max_length: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        self.batch_size = batch_size
        self.max_length = max_length

    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predict sentiment for a batch of texts."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                labels = predictions.argmax(dim=-1)
        
        # Convert to labels
        return ["Positive" if label == 1 else "Negative" for label in labels.cpu().numpy()]

    def process_dataset(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process dataset with sentiment analysis."""
        print("Starting sentiment analysis...")
        
        # Create dataset and dataloader
        dataset = TextDataset(df[text_column].tolist())
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid overhead
            pin_memory=True  # Enable pinned memory for faster data transfer to GPU
        )
        
        sentiments = []
        
        # Process in batches with progress bar
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Predict sentiments
            batch_sentiments = self.predict_batch(batch)
            sentiments.extend(batch_sentiments)
        
        # Add results to dataframe
        df['sentiment'] = sentiments
        
        return df

def main():
    # Adjust batch size based on your GPU memory
    batch_size = 256  
    analyzer = SentimentAnalyzer(batch_size=batch_size)
    
    # Load your DataFrame here
    df = pd.read_csv("professors_with_disciplines.csv")
    
    # Process the dataset
    df = analyzer.process_dataset(df, 'rating_comment')
    
    # Save results
    df.to_csv("professors_with_SA_labelling.csv.csv", index=False)
    
    # Clean up
    del analyzer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
