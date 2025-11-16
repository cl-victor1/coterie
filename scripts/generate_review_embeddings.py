"""
Generate review embeddings (supports resume from interruption)
Use OpenAI text-embedding-3-large model to generate vector representations for all reviews
Save every 100 entries, supports continuing from interruption point
"""

import json
import os
from openai import OpenAI
from tqdm import tqdm
import time
from datetime import datetime
from dotenv import load_dotenv

def load_reviews(json_path):
    """Load review data"""
    print("Loading review data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_embedding_text(review):
    """Prepare text for embedding"""
    title = review.get('title', '').strip()
    content = review.get('content', '').strip()
    
    # Combine title and content
    if title and content:
        text = f"{title}. {content}"
    elif content:
        text = content
    elif title:
        text = title
    else:
        text = ""
    
    return text

def load_progress(progress_path):
    """Load progress file"""
    if os.path.exists(progress_path):
        print(f"Found progress file: {progress_path}")
        with open(progress_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Completed: {len(data['completed_embeddings'])}/{data['total_reviews']}")
        return data
    return None

def save_progress(progress_path, progress_data):
    """Save progress"""
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

def format_review_data(review, embedding):
    """Format single review data"""
    # Safely get custom_fields
    custom_fields = review.get('custom_fields') or {}
    
    return {
        "id": review['id'],
        "title": review.get('title', ''),
        "content": review.get('content', ''),
        "score": review.get('score', 0),
        "sentiment": review.get('sentiment', 0),
        "embedding": embedding,
        "metadata": {
            "verified_buyer": review.get('verified_buyer', False),
            "created_at": review.get('created_at', ''),
            "size": custom_fields.get('--116326', {}).get('value', 'N/A') if isinstance(custom_fields.get('--116326'), dict) else 'N/A',
            "age": custom_fields.get('--116327', {}).get('value', 'N/A') if isinstance(custom_fields.get('--116327'), dict) else 'N/A',
        }
    }

def generate_embeddings_incremental(client, reviews, progress_path, batch_size=100):
    """Generate embeddings incrementally, supports resume from interruption"""
    
    # Load or initialize progress
    progress_data = load_progress(progress_path)
    
    if progress_data is None:
        # Initialize progress data
        progress_data = {
            "total_reviews": len(reviews),
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072,
            "started_at": datetime.now().isoformat(),
            "completed_embeddings": [],
            "failed_ids": []
        }
    
    # Get set of completed IDs
    completed_ids = {item['id'] for item in progress_data['completed_embeddings']}
    
    # Filter out completed reviews
    remaining_reviews = [r for r in reviews if r['id'] not in completed_ids]
    
    if not remaining_reviews:
        print("All embeddings have been generated!")
        return progress_data
    
    print(f"\nReviews to process: {len(remaining_reviews)}")
    print(f"Starting incremental embedding generation (batch size: {batch_size})...")
    
    # Process in batches
    for i in tqdm(range(0, len(remaining_reviews), batch_size)):
        batch_reviews = remaining_reviews[i:i + batch_size]
        
        # Prepare texts
        texts = [prepare_embedding_text(r) for r in batch_reviews]
        
        # Filter empty texts
        valid_indices = [idx for idx, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[idx] for idx in valid_indices]
        
        if not valid_texts:
            # If entire batch is empty text, record as failed
            for review in batch_reviews:
                progress_data['failed_ids'].append(review['id'])
            continue
        
        try:
            # Call OpenAI API
            response = client.embeddings.create(
                input=valid_texts,
                model="text-embedding-3-large"
            )
            
            # Extract embeddings and save
            for idx, review in enumerate(batch_reviews):
                if idx in valid_indices:
                    # Find corresponding embedding
                    valid_idx = valid_indices.index(idx)
                    embedding = response.data[valid_idx].embedding
                    
                    # Format and add to completed list
                    review_data = format_review_data(review, embedding)
                    progress_data['completed_embeddings'].append(review_data)
                else:
                    # Empty text, record as failed
                    progress_data['failed_ids'].append(review['id'])
            
            # Save progress after each batch
            save_progress(progress_path, progress_data)
            
            # Avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nBatch {i//batch_size + 1} error: {str(e)}")
            print("Waiting 5 seconds before retry...")
            time.sleep(5)
            
            # Retry once
            try:
                response = client.embeddings.create(
                    input=valid_texts,
                    model="text-embedding-3-large"
                )
                
                for idx, review in enumerate(batch_reviews):
                    if idx in valid_indices:
                        valid_idx = valid_indices.index(idx)
                        embedding = response.data[valid_idx].embedding
                        review_data = format_review_data(review, embedding)
                        progress_data['completed_embeddings'].append(review_data)
                    else:
                        progress_data['failed_ids'].append(review['id'])
                
                # Save progress
                save_progress(progress_path, progress_data)
                
            except Exception as e2:
                print(f"Retry failed: {str(e2)}")
                # Record entire batch as failed
                for review in batch_reviews:
                    progress_data['failed_ids'].append(review['id'])
                # Save progress and exit
                save_progress(progress_path, progress_data)
                print(f"Progress saved to: {progress_path}")
                print("Please fix the error and rerun the script to continue")
                return progress_data
    
    # Update completion time
    progress_data['completed_at'] = datetime.now().isoformat()
    save_progress(progress_path, progress_data)
    
    return progress_data

def main():
    # Initialize OpenAI client
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # File paths
    input_path = "/Users/victor_official/AI personas/evaluation/the_diaper_reviews.json"
    progress_path = "/Users/victor_official/AI personas/evaluation/review_embeddings_progress.json"
    
    # Load reviews
    data = load_reviews(input_path)
    reviews = data['reviews']
    
    print(f"Total reviews: {len(reviews)}")
    
    # Generate embeddings incrementally
    progress_data = generate_embeddings_incremental(client, reviews, progress_path, batch_size=100)
    
    # Generate final output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/victor_official/AI personas/evaluation/review_embeddings_{timestamp}.json"
    
    print("\nPreparing final output data...")
    output_data = {
        "metadata": {
            "source_file": input_path,
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072,
            "total_reviews": len(reviews),
            "successful_embeddings": len(progress_data['completed_embeddings']),
            "failed_embeddings": len(progress_data['failed_ids']),
            "started_at": progress_data.get('started_at'),
            "completed_at": progress_data.get('completed_at', datetime.now().isoformat()),
        },
        "embeddings": progress_data['completed_embeddings']
    }
    
    # Save final results
    print(f"\nSaving final results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Output statistics
    print("\n" + "="*60)
    print("Complete! Statistics:")
    print("="*60)
    print(f"Total reviews: {output_data['metadata']['total_reviews']}")
    print(f"Successfully generated embeddings: {output_data['metadata']['successful_embeddings']}")
    print(f"Failed: {output_data['metadata']['failed_embeddings']}")
    if progress_data['failed_ids']:
        print(f"Failed IDs: {progress_data['failed_ids'][:10]}{'...' if len(progress_data['failed_ids']) > 10 else ''}")
    print(f"Output file: {output_path}")
    print(f"Progress file: {progress_path}")
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    print("="*60)
    
    # Note about progress file
    print("\nNote: Progress file has been retained, delete manually if regeneration is needed")
    print(f"Delete command: rm '{progress_path}'")

if __name__ == "__main__":
    main()
