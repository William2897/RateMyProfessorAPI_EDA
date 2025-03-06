import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models import LdaMulticore, CoherenceModel
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

def read_data_in_chunks(file_path, chunk_size=50000):
    """Read CSV file in chunks to prevent memory overload"""
    return pd.read_csv(file_path, chunksize=chunk_size)

def convert_to_list(text):
    """Safely convert text to list if needed"""
    if isinstance(text, str):
        try:
            return eval(text)
        except:
            return []
    return text if isinstance(text, list) else []

def create_dictionary_from_chunks(file_path, chunk_size=50000):
    """Create dictionary incrementally from text chunks"""
    dictionary = corpora.Dictionary()
    chunks = read_data_in_chunks(file_path, chunk_size)
    
    for chunk in tqdm(chunks, desc="Building dictionary"):
        texts = chunk['processed_comment'].apply(convert_to_list).tolist()
        texts = [text for text in texts if isinstance(text, list) and len(text) > 0]
        if texts:
            dictionary.add_documents(texts)
        
        # Clear memory
        del texts
        gc.collect()
    
    # Filter extremes after processing all documents
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    return dictionary

def process_chunk_to_corpus(chunk, dictionary):
    """Convert a chunk of texts to corpus format"""
    texts = chunk['processed_comment'].apply(convert_to_list).tolist()
    texts = [text for text in texts if isinstance(text, list) and len(text) > 0]
    corpus = [dictionary.doc2bow(text) for text in texts]
    return [doc for doc in corpus if len(doc) > 0], texts

def compute_coherence_scores(dictionary, file_path, num_topics_range, chunk_size=50000):
    """Compute coherence scores using a subset of data"""
    # Use only first chunk for coherence computation
    first_chunk = next(read_data_in_chunks(file_path, chunk_size))
    corpus, texts = process_chunk_to_corpus(first_chunk, dictionary)
    
    # Use a smaller subset for coherence calculation
    subset_size = min(10000, len(corpus))
    corpus_subset = corpus[:subset_size]
    texts_subset = texts[:subset_size]
    
    coherence_values = []
    
    for num_topics in tqdm(num_topics_range, desc="Computing coherence scores"):
        try:
            model = LdaMulticore(
                corpus=corpus_subset,
                num_topics=num_topics,
                id2word=dictionary,
                random_state=42,
                passes=2,  # Reduced passes for faster processing
                workers=7,  # Using most of available cores
                chunksize=2000
            )
            
            coherencemodel = CoherenceModel(
                model=model,
                texts=texts_subset,
                dictionary=dictionary,
                coherence='c_v',
                processes=7
            )
            
            coherence = coherencemodel.get_coherence()
            coherence_values.append(coherence)
            
            # Clear memory
            del model, coherencemodel
            gc.collect()
            
        except Exception as e:
            print(f"Error for num_topics={num_topics}: {e}")
            coherence_values.append(None)
    
    return coherence_values

def train_final_model(optimal_num_topics, dictionary, file_path, chunk_size=50000):
    """Train the final LDA model using chunks"""
    model = LdaMulticore(
        id2word=dictionary,
        num_topics=optimal_num_topics,
        random_state=42,
        passes=10,
        workers=7,
        chunksize=2000
    )
    
    # Train model incrementally
    chunks = read_data_in_chunks(file_path, chunk_size)
    for chunk in tqdm(chunks, desc="Training final model"):
        corpus, _ = process_chunk_to_corpus(chunk, dictionary)
        if corpus:
            model.update(corpus)
        
        # Clear memory
        del corpus
        gc.collect()
    
    return model

def process_and_save_results(model, dictionary, file_path, output_path, chunk_size=50000):
    """Process results and save in chunks"""
    chunks = read_data_in_chunks(file_path, chunk_size)
    first_chunk = True
    
    for chunk_index, chunk in enumerate(tqdm(chunks, desc="Processing results")):
        corpus, _ = process_chunk_to_corpus(chunk, dictionary)
        
        # Get dominant topics
        dominant_topics = []
        for doc in corpus:
            topic_dist = model.get_document_topics(doc)
            dominant_topic = max(topic_dist, key=lambda x: x[1]) if topic_dist else (-1, 0)
            topic_keywords = ", ".join([word for word, _ in model.show_topic(dominant_topic[0])])
            dominant_topics.append((dominant_topic[0], dominant_topic[1], topic_keywords))
        
        # Create results DataFrame
        df_topics = pd.DataFrame(dominant_topics, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
        results = pd.concat([chunk.reset_index(drop=True), df_topics], axis=1)
        
        # Save chunk
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        results.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False
        
        # Clear memory
        del corpus, dominant_topics, df_topics, results
        gc.collect()

def main_pipeline(input_file, output_file):
    """Main processing pipeline with memory optimization"""
    # Step 1: Create dictionary
    print("Creating dictionary...")
    dictionary = create_dictionary_from_chunks(input_file)
    print(f"Dictionary created with {len(dictionary)} terms")
    
    # Step 2: Find optimal number of topics
    print("Computing coherence scores...")
    topic_range = range(3, 12, 1)
    coherence_values = compute_coherence_scores(dictionary, input_file, topic_range)
    
    # Plot coherence scores
    if any(cv is not None for cv in coherence_values):
        plt.figure(figsize=(10, 6))
        plt.plot(topic_range, coherence_values, marker='o')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Coherence Scores by Number of Topics')
        plt.grid(True)
        plt.show()
        
        optimal_num_topics = topic_range[np.argmax(coherence_values)]
        print(f"Optimal number of topics: {optimal_num_topics}")
        
        # Step 3: Train final model
        print("Training final model...")
        optimal_model = train_final_model(optimal_num_topics, dictionary, input_file)
        
        # Save model and dictionary
        print("Saving model and dictionary...")
        with open('lda_model_optimized.pkl', 'wb') as f:
            pickle.dump(optimal_model, f)
        with open('dictionary_optimized.pkl', 'wb') as f:
            pickle.dump(dictionary, f)
        
        # Step 4: Process and save results
        print("Processing and saving results...")
        process_and_save_results(optimal_model, dictionary, input_file, output_file)
        
        return optimal_model, dictionary
    else:
        print("No valid coherence values computed")
        return None, None

if __name__ == "__main__":
    input_file = "RMP SET.csv"
    output_file = "topic_results_optimized.csv"
    optimal_model, dictionary = main_pipeline(input_file, output_file)