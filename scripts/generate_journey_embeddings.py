"""
Generate journey embeddings
Use OpenAI text-embedding-3-large model to generate vector representations for test journeys
Only embed journey, friction_points and evaluation
"""

import json
import os
import sys
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv


def prepare_journey_text(persona_data):
    """Prepare journey text for embedding
    
    Combine key information from journey, friction_points and evaluation into a complete description
    """
    persona_name = persona_data.get("persona_name", "Unknown")
    task_completed = "completed" if persona_data.get("task_completed") else "abandoned"
    
    # Journey information
    journey_steps = []
    for step in persona_data.get("journey", []):
        step_num = step.get("step", 0)
        action = step.get("action", "")
        reasoning = step.get("reasoning", "")
        if reasoning:
            journey_steps.append(f"Step {step_num} ({action}): {reasoning}")
    
    journey_text = "User Journey:\n" + "\n".join(journey_steps)
    
    # Friction points information
    friction_text = "Friction Points:\n"
    page_analysis = persona_data.get("friction_points", {}).get("page_analysis", [])
    for idx, friction in enumerate(page_analysis, 1):
        desc = friction.get("description", "")
        severity = friction.get("severity", "")
        friction_text += f"{idx}. [{severity}] {desc}\n"
    
    # Evaluation information
    evaluation = persona_data.get("evaluation", {})
    
    completion_reason = f"Completion Reason: {evaluation.get('completion_reason', '')}"
    
    major_frictions = evaluation.get("major_friction_points", [])
    major_friction_text = "Major Friction Points:\n" + "\n".join([f"- {fp}" for fp in major_frictions])
    
    positive_elements = evaluation.get("positive_elements", [])
    positive_text = "Positive Elements:\n" + "\n".join([f"- {pe}" for pe in positive_elements])
    
    satisfaction = f"Satisfaction Score: {evaluation.get('persona_satisfaction', 'N/A')}/10"
    
    recommendations = evaluation.get("recommendations", [])
    recommendations_text = "Recommendations:\n" + "\n".join([f"- {rec}" for rec in recommendations])
    
    # Combine all text
    full_text = f"""
Persona: {persona_name}
Task Status: {task_completed}

{journey_text}

{friction_text}

{completion_reason}

{major_friction_text}

{positive_text}

{satisfaction}

{recommendations_text}
    """.strip()
    
    return full_text


def generate_single_embedding(client, text, model="text-embedding-3-large"):
    """Generate embedding for a single text"""
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None


def main():
    # Initialize OpenAI client
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Input file path
    input_path = "/Users/victor_official/AI personas/output/test_20251115_124409.json"
    
    # Output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/victor_official/AI personas/evaluation/journey_embeddings_{timestamp}.json"
    
    print("="*60)
    print("Starting Journey Embeddings Generation")
    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Model: text-embedding-3-large")
    print(f"Dimension: 3072")
    print("="*60)
    
    # Read test data
    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    personas_data = test_data.get("personas", [])
    print(f"Total Journeys: {len(personas_data)}")
    
    # Prepare output data
    output_data = {
        "metadata": {
            "source_file": input_path,
            "test_timestamp": test_data.get("test_metadata", {}).get("timestamp", ""),
            "task": test_data.get("test_metadata", {}).get("task", ""),
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072,
            "total_journeys": len(personas_data),
            "generated_at": datetime.now().isoformat(),
        },
        "journeys": []
    }
    
    # Generate embedding for each journey
    for idx, persona_data in enumerate(personas_data, 1):
        persona_name = persona_data.get("persona_name", f"Persona {idx}")
        print(f"\nProcessing {idx}/{len(personas_data)}: {persona_name}")
        
        # Prepare text
        text = prepare_journey_text(persona_data)
        print(f"  Text length: {len(text)} characters")
        
        # Generate embedding
        embedding = generate_single_embedding(client, text)
        
        if embedding:
            print(f"  ✓ Successfully generated embedding (dimension: {len(embedding)})")
        else:
            print(f"  ✗ Generation failed")
        
        # Save journey information and embedding
        journey_data = {
            "persona_name": persona_name,
            "test_timestamp": persona_data.get("test_timestamp", ""),
            "task_completed": persona_data.get("task_completed", False),
            "abandoned": persona_data.get("abandoned", False),
            "entry_point": persona_data.get("entry_point", ""),
            "device": persona_data.get("device", ""),
            "journey_steps": len(persona_data.get("journey", [])),
            "friction_count": len(persona_data.get("friction_points", {}).get("page_analysis", [])),
            "persona_satisfaction": persona_data.get("evaluation", {}).get("persona_satisfaction", None),
            "embedding_text": text,
            "embedding": embedding,
            "journey": persona_data.get("journey", []),
            "friction_points": persona_data.get("friction_points", {}),
            "evaluation": persona_data.get("evaluation", {})
        }
        
        output_data["journeys"].append(journey_data)
    
    # Update statistics
    successful = sum(1 for j in output_data["journeys"] if j["embedding"] is not None)
    output_data["metadata"]["successful_embeddings"] = successful
    output_data["metadata"]["failed_embeddings"] = len(personas_data) - successful
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Output statistics
    print("\n" + "="*60)
    print("Complete! Statistics:")
    print("="*60)
    print(f"Total Journeys: {output_data['metadata']['total_journeys']}")
    print(f"Successfully generated embeddings: {output_data['metadata']['successful_embeddings']}")
    print(f"Failed: {output_data['metadata']['failed_embeddings']}")
    print(f"Output file: {output_path}")
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    print("="*60)
    
    # Print detailed information for each journey
    print("\n" + "="*60)
    print("Journey Details:")
    print("="*60)
    for journey_data in output_data["journeys"]:
        print(f"\n{journey_data['persona_name']}")
        print(f"  Device: {journey_data['device']}")
        print(f"  Task completed: {'Yes' if journey_data['task_completed'] else 'No'}")
        print(f"  Journey steps: {journey_data['journey_steps']}")
        print(f"  Friction points: {journey_data['friction_count']}")
        print(f"  Satisfaction score: {journey_data['persona_satisfaction']}")
        embedding_status = "✓" if journey_data['embedding'] else "✗"
        print(f"  Embedding: {embedding_status}")


if __name__ == "__main__":
    main()

