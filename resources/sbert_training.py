from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

# Load your labeled data
df = pd.read_csv("labeled_resume_job_pairs.csv")

# Convert rows into SBERT training format
train_examples = [
    InputExample(texts=[row['resume_text'], row['job_text']], label=float(row['label']))
    for _, row in df.iterrows()
]

# Load a pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the training loss
train_loss = losses.CosineSimilarityLoss(model=model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,  # Increase to 3–5 for better performance
    warmup_steps=100
)

# Save the fine-tuned model
model_output_path = "models/sbert-finetuned-resumes"
os.makedirs(model_output_path, exist_ok=True)
model.save(model_output_path)

print(f"Model saved to {model_output_path}")