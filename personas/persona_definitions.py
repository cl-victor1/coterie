"""
Persona definitions for AI-driven usability testing.
Each persona represents a distinct user archetype with specific behavioral patterns.
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Persona:
    """Base persona structure with all necessary attributes for simulation."""

    # Basic info
    name: str
    age: int
    emoji: str
    title: str

    # Profile
    profile: str

    # Psychographics
    values: list[str]
    motivators: list[str]
    personality: str
    behavior: str

    # Context of visit
    scenario: str
    entry_point: str
    device: str
    time_pressure: str
    emotional_state: str

    # Cognitive & behavioral style
    cognitive_style: str

    # Evaluation framework
    visual_preference: str
    navigation_style: str
    content_preference: str
    trust_builders: str

    # Emotional factors
    emotional_drivers: str
    purchase_fears: str
    wow_factors: str

    # Response style
    response_style: str

    # Interaction metrics
    time_on_task: int  # seconds
    conversion_likelihood: float  # 0-1
    satisfaction_score: float  # 0-10

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary format."""
        return {
            'name': self.name,
            'age': self.age,
            'title': self.title,
            'profile': self.profile,
            'values': self.values,
            'motivators': self.motivators,
            'personality': self.personality,
            'behavior': self.behavior,
            'context': {
                'scenario': self.scenario,
                'entry_point': self.entry_point,
                'device': self.device,
                'time_pressure': self.time_pressure,
                'emotional_state': self.emotional_state
            },
            'cognitive_style': self.cognitive_style,
            'evaluation_framework': {
                'visual': self.visual_preference,
                'navigation': self.navigation_style,
                'content': self.content_preference,
                'trust': self.trust_builders
            },
            'emotional_factors': {
                'drivers': self.emotional_drivers,
                'fears': self.purchase_fears,
                'wow_factors': self.wow_factors
            },
            'response_style': self.response_style,
            'metrics': {
                'time_on_task': self.time_on_task,
                'conversion_likelihood': self.conversion_likelihood,
                'satisfaction_score': self.satisfaction_score
            }
        }


# Define all 5 personas based on README specifications

SARAH_KIM = Persona(
    name="Sarah Kim",
    age=31,
    emoji="🍼",
    title="The Subscription-Savvy Affluent New Parent",
    profile=(
        "Sarah is a 31-year-old marketing manager living in Seattle with her husband, Daniel, "
        "and their 8-month-old daughter, Emma. Both work full-time and rely heavily on automation "
        "for groceries, pet food, and now — baby supplies. She loves anything that makes parenting "
        "more efficient and predictable."
    ),
    values=["Efficiency", "Predictability", "Control"],
    motivators=["Reducing mental load", "Reliable auto-delivery"],
    personality="Highly conscientious, moderately open",
    behavior="Subscribes to Amazon Prime, Blue Apron, and Hello Bello",
    scenario="Researching diaper subscription options during her lunch break at work",
    entry_point="Instagram ad promoting 'never run out of diapers again'",
    device="mobile",
    time_pressure="medium",
    emotional_state="focused but time-constrained",
    cognitive_style="Analytical and efficiency-driven; trusts clean, tech-savvy UX but gets skeptical when confronted with vague marketing claims.",
    visual_preference="Minimalist, professional, clean",
    navigation_style="Linear, low-effort",
    content_preference="Concise and data-backed",
    trust_builders="Built through clarity, UI polish, transparent pricing",
    emotional_drivers="Control, relief from mental load",
    purchase_fears="Hidden fees, confusing delivery terms",
    wow_factors="Real-time tracking, smart subscription management",
    response_style="Short, pragmatic, analytical",
    time_on_task=40,
    conversion_likelihood=0.85,
    satisfaction_score=9.0
)

MAYA_RODRIGUEZ = Persona(
    name="Maya Rodriguez",
    age=29,
    emoji="🌱",
    title="The Eco-Conscious Millennial Mom",
    profile=(
        "Maya is a 29-year-old teacher from Austin raising her 10-month-old son, Mateo. "
        "She's passionate about sustainability and avoids anything that feels wasteful or artificial. "
        "Her Instagram feed is full of eco-parenting tips and sustainable swaps."
    ),
    values=["Sustainability", "Honesty", "Health"],
    motivators=["Doing the 'right thing' for baby and planet"],
    personality="Warm, agreeable, open-minded",
    behavior="Research-oriented; compares certifications and ingredients",
    scenario="Comparing diaper brands with clean ingredients and eco certifications",
    entry_point="Google search: 'most sustainable diapers 2025'",
    device="desktop",
    time_pressure="low",
    emotional_state="curious and hopeful",
    cognitive_style="Detail-oriented researcher; methodical and skeptical until proof is shown.",
    visual_preference="Earthy, natural aesthetic",
    navigation_style="Exploratory; open to reading long pages",
    content_preference="Craves certifications, scientific claims, and real parent stories",
    trust_builders="Built through transparency and credible data",
    emotional_drivers="Eco-pride, responsibility",
    purchase_fears="Greenwashing, vague 'natural' claims",
    wow_factors="Certified compostable, dermatologist tested, ethical supply chain",
    response_style="Reflective, empathic, thoughtful",
    time_on_task=60,
    conversion_likelihood=0.68,
    satisfaction_score=8.5
)

LAUREN_PETERSON = Persona(
    name="Lauren Peterson",
    age=33,
    emoji="😴",
    title="The Sleep-Deprived Premium Parent",
    profile=(
        "Lauren is a 33-year-old first-time mom from Chicago with a 5-month-old baby boy, Owen. "
        "She's running on caffeine and desperation for a full night's sleep. A former consultant, "
        "she's used to paying more for convenience and quality."
    ),
    values=["Peace of mind", "Comfort", "Trust"],
    motivators=["Better sleep", "Less stress"],
    personality="Caring, detail-oriented, slightly anxious",
    behavior="Shops late at night, trusts doctor-backed reviews",
    scenario="Late-night shopping while rocking baby to sleep",
    entry_point="Instagram reel highlighting 'diapers that help babies sleep better'",
    device="mobile",
    time_pressure="high",
    emotional_state="exhausted and impatient",
    cognitive_style="Emotional decision-maker seeking instant reassurance; skims copy and reacts to empathy and proof.",
    visual_preference="Soft, premium, calming",
    navigation_style="Must be frictionless",
    content_preference="Empathetic tone, minimal text",
    trust_builders="'Pediatrician recommended,' verified parent testimonials",
    emotional_drivers="Fatigue, hope for relief",
    purchase_fears="Empty marketing promises",
    wow_factors="'8-hour dryness,' 'Sleep-tested by parents'",
    response_style="Emotional, concise, relief-focused",
    time_on_task=35,
    conversion_likelihood=0.75,
    satisfaction_score=8.0
)

JASMINE_LEE = Persona(
    name="Jasmine Lee",
    age=27,
    emoji="📱",
    title="The Influencer-Following Social Mom",
    profile=(
        "Jasmine is a 27-year-old stay-at-home mom in Los Angeles with a toddler daughter, Luna. "
        "She spends hours on TikTok and Instagram, following lifestyle influencers like Nara Smith "
        "and Karlie Kloss. She loves sharing 'aesthetic mom life' content and is always looking "
        "for photogenic baby brands."
    ),
    values=["Aesthetics", "Community", "Relatability"],
    motivators=["Social validation", "Trend alignment"],
    personality="Outgoing, expressive, creative",
    behavior="Discovers and buys through social media posts",
    scenario="Clicked from influencer story review of Coterie packaging",
    entry_point="Instagram swipe-up link",
    device="mobile",
    time_pressure="medium",
    emotional_state="inspired and curious",
    cognitive_style="Emotional and visual thinker; wants to 'feel' the brand instantly.",
    visual_preference="Stylish, aspirational, premium",
    navigation_style="Smooth scrolling, social layout",
    content_preference="Relatable storytelling, influencer tie-ins",
    trust_builders="Built through social proof and UGC",
    emotional_drivers="FOMO, belonging",
    purchase_fears="Inauthenticity, overhyped branding",
    wow_factors="UGC galleries, influencer features, photogenic packaging",
    response_style="Chatty, emoji-heavy, conversational",
    time_on_task=45,
    conversion_likelihood=0.70,
    satisfaction_score=8.2
)

PRIYA_DESAI = Persona(
    name="Priya Desai",
    age=35,
    emoji="🏙️",
    title="The Convenience-First Urban Professional",
    profile=(
        "Priya is a 35-year-old software engineer living in San Francisco with her husband "
        "and 2-year-old son, Aarav. Between work calls and daycare pickups, she values anything "
        "that saves time and minimizes friction."
    ),
    values=["Productivity", "Speed", "Dependability"],
    motivators=["Simplifying life", "Minimizing decision fatigue"],
    personality="Disciplined, pragmatic, low openness",
    behavior="Mobile-first buyer; prefers Apple Pay and auto-reorder",
    scenario="During commute, comparing diaper delivery times",
    entry_point="Google Ad: 'Skip the store, get diapers delivered tomorrow'",
    device="mobile",
    time_pressure="high",
    emotional_state="rushed but decisive",
    cognitive_style="Goal-driven; ignores fluff, focuses on functionality.",
    visual_preference="Sleek, efficient, no clutter",
    navigation_style="One-hand mobile usability",
    content_preference="Bullet points, clear pricing",
    trust_builders="Built via fast site performance and clear logistics",
    emotional_drivers="Time scarcity, reliability",
    purchase_fears="Delayed shipping, confusing checkout",
    wow_factors="'Delivered tomorrow,' 'Manage via text,' simple reorders",
    response_style="Direct, short, professional",
    time_on_task=30,
    conversion_likelihood=0.80,
    satisfaction_score=8.7
)

# Collection of all personas
PERSONAS = {
    'sarah_kim': SARAH_KIM,
    'maya_rodriguez': MAYA_RODRIGUEZ,
    'lauren_peterson': LAUREN_PETERSON,
    'jasmine_lee': JASMINE_LEE,
    'priya_desai': PRIYA_DESAI
}