"""
Generate persona embeddings
Use OpenAI text-embedding-3-large model to generate vector representations for all personas
"""

import json
import os
import sys
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# 直接导入persona_definitions模块以避免循环导入
personas_module_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'personas',
    'persona_definitions.py'
)

# 动态加载模块
import importlib.util
spec = importlib.util.spec_from_file_location("persona_definitions", personas_module_path)
persona_definitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(persona_definitions)

PERSONAS = persona_definitions.PERSONAS


def prepare_persona_text(persona):
    """Prepare persona text for embedding
    
    Combine key persona information into a complete description
    """
    # Basic information
    basic_info = f"{persona.name}, {persona.age} years old, {persona.title}"
    
    # Detailed profile
    profile = persona.profile
    
    # Psychological characteristics
    values = "Values: " + ", ".join(persona.values)
    motivators = "Motivators: " + ", ".join(persona.motivators)
    personality = f"Personality: {persona.personality}"
    behavior = f"Behavior: {persona.behavior}"
    
    # Usage scenario
    scenario = f"Scenario: {persona.scenario}"
    entry_point = f"Entry Point: {persona.entry_point}"
    device = f"Device: {persona.device}"
    time_pressure = f"Time Pressure: {persona.time_pressure}"
    emotional_state = f"Emotional State: {persona.emotional_state}"
    
    # Cognitive style
    cognitive = f"Cognitive Style: {persona.cognitive_style}"
    
    # Preferences and needs
    visual = f"Visual Preference: {persona.visual_preference}"
    navigation = f"Navigation Style: {persona.navigation_style}"
    content = f"Content Preference: {persona.content_preference}"
    trust = f"Trust Building: {persona.trust_builders}"
    
    # Emotional factors
    drivers = f"Emotional Drivers: {persona.emotional_drivers}"
    fears = f"Purchase Fears: {persona.purchase_fears}"
    wow = f"Wow Factors: {persona.wow_factors}"
    
    # Combine all text
    full_text = f"""
{basic_info}

{profile}

{values}
{motivators}
{personality}
{behavior}

{scenario}
{entry_point}
{device}
{time_pressure}
{emotional_state}

{cognitive}

{visual}
{navigation}
{content}
{trust}

{drivers}
{fears}
{wow}
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
    
    # Output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/victor_official/AI personas/evaluation/persona_embeddings_{timestamp}.json"
    
    print("="*60)
    print("Starting Persona Embeddings Generation")
    print("="*60)
    print(f"Total Personas: {len(PERSONAS)}")
    print(f"Model: text-embedding-3-large")
    print(f"Dimension: 3072")
    print("="*60)
    
    # Prepare output data
    output_data = {
        "metadata": {
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072,
            "total_personas": len(PERSONAS),
            "generated_at": datetime.now().isoformat(),
        },
        "personas": []
    }
    
    # Generate embedding for each persona
    for persona_key, persona in PERSONAS.items():
        print(f"\nProcessing: {persona.name} ({persona.emoji} {persona.title})")
        
        # Prepare text
        text = prepare_persona_text(persona)
        print(f"  Text length: {len(text)} characters")
        
        # Generate embedding
        embedding = generate_single_embedding(client, text)
        
        if embedding:
            print(f"  ✓ Successfully generated embedding (dimension: {len(embedding)})")
        else:
            print(f"  ✗ Generation failed")
        
        # Save persona information and embedding
        persona_data = {
            "key": persona_key,
            "name": persona.name,
            "age": persona.age,
            "emoji": persona.emoji,
            "title": persona.title,
            "profile": persona.profile,
            "values": persona.values,
            "motivators": persona.motivators,
            "personality": persona.personality,
            "behavior": persona.behavior,
            "context": {
                "scenario": persona.scenario,
                "entry_point": persona.entry_point,
                "device": persona.device,
                "time_pressure": persona.time_pressure,
                "emotional_state": persona.emotional_state
            },
            "cognitive_style": persona.cognitive_style,
            "evaluation_framework": {
                "visual_preference": persona.visual_preference,
                "navigation_style": persona.navigation_style,
                "content_preference": persona.content_preference,
                "trust_builders": persona.trust_builders
            },
            "emotional_factors": {
                "emotional_drivers": persona.emotional_drivers,
                "purchase_fears": persona.purchase_fears,
                "wow_factors": persona.wow_factors
            },
            "response_style": persona.response_style,
            "metrics": {
                "time_on_task": persona.time_on_task,
                "conversion_likelihood": persona.conversion_likelihood,
                "satisfaction_score": persona.satisfaction_score
            },
            "embedding_text": text,
            "embedding": embedding
        }
        
        output_data["personas"].append(persona_data)
    
    # Update statistics
    successful = sum(1 for p in output_data["personas"] if p["embedding"] is not None)
    output_data["metadata"]["successful_embeddings"] = successful
    output_data["metadata"]["failed_embeddings"] = len(PERSONAS) - successful
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Output statistics
    print("\n" + "="*60)
    print("Complete! Statistics:")
    print("="*60)
    print(f"Total Personas: {output_data['metadata']['total_personas']}")
    print(f"Successfully generated embeddings: {output_data['metadata']['successful_embeddings']}")
    print(f"Failed: {output_data['metadata']['failed_embeddings']}")
    print(f"Output file: {output_path}")
    
    # Calculate file size
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"File size: {file_size_kb:.2f} KB")
    print("="*60)
    
    # Print detailed information for each persona
    print("\n" + "="*60)
    print("Persona Details:")
    print("="*60)
    for persona_data in output_data["personas"]:
        print(f"\n{persona_data['emoji']} {persona_data['name']}")
        print(f"  Title: {persona_data['title']}")
        print(f"  Age: {persona_data['age']}")
        print(f"  Device: {persona_data['context']['device']}")
        print(f"  Time Pressure: {persona_data['context']['time_pressure']}")
        print(f"  Emotional State: {persona_data['context']['emotional_state']}")
        print(f"  Conversion Likelihood: {persona_data['metrics']['conversion_likelihood']}")
        print(f"  Satisfaction Score: {persona_data['metrics']['satisfaction_score']}")
        embedding_status = "✓" if persona_data['embedding'] else "✗"
        print(f"  Embedding: {embedding_status}")


if __name__ == "__main__":
    main()

