#!/usr/bin/env python3
"""
Medical Accuracy Scoring Framework for MedGemma Impact Challenge
Evaluates health educator responses on medical accuracy, accessibility, and quality dimensions.
"""

import json
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MedicalAccuracyScoringFramework:
    """Evaluates MedGemma responses using pattern-based analysis."""

    # Define scoring rubrics
    SCORING_RUBRICS = {
        "Medical Accuracy": {
            "5": "All claims are scientifically correct and well-supported",
            "4": "Most claims are accurate with minor oversimplifications",
            "3": "Generally accurate but contains some questionable statements",
            "2": "Contains significant inaccuracies or misleading information",
            "1": "Highly inaccurate or dangerous medical claims"
        },
        "Patient Accessibility": {
            "5": "Excellent - clear analogies, simple language, jargon explained",
            "4": "Good - mostly clear with adequate explanations",
            "3": "Acceptable - some complex terms without sufficient explanation",
            "2": "Poor - heavy jargon, difficult for layperson to understand",
            "1": "Very Poor - overly technical, inaccessible language"
        },
        "Analogy Quality": {
            "5": "Exceptional analogies that clarify difficult concepts",
            "4": "Good analogies that help explain the topic",
            "3": "Adequate analogies but could be clearer",
            "2": "Weak or confusing analogies",
            "1": "No analogies or misleading ones"
        },
        "Actionability": {
            "5": "Specific, practical, prioritized tips for the reader",
            "4": "Good actionable advice with clear steps",
            "3": "Some actionable content but lacks specificity",
            "2": "Minimal actionable advice",
            "1": "No practical tips or advice"
        },
        "Safety/Disclaimers": {
            "5": "Strong disclaimers, clear medical advice limits",
            "4": "Appropriate disclaimers present",
            "3": "Basic disclaimer but could be stronger",
            "2": "Minimal or weak disclaimers",
            "1": "No disclaimers or warnings"
        },
        "Completeness": {
            "5": "Thoroughly covers all key aspects of the topic",
            "4": "Covers most key aspects comprehensively",
            "3": "Covers main aspects but lacks some depth",
            "2": "Incomplete coverage of the topic",
            "1": "Superficial or incomplete coverage"
        }
    }

    # Keyword and pattern definitions
    ANALOGY_INDICATORS = [
        r'\blike\b', r'\bthink of\b', r'\bimagine\b', r'\bsimilar to\b',
        r'\bas if\b', r'\banalogous\b', r'\bcomparable\b', r'\bresembles\b',
        r'\bmetaphor\b', r'\bas a\b', r'\brecycling\b', r'\bgarbage\b',
        r'\bfactory\b', r'\btruck\b', r'\bfire\b', r'\bspark\b', r'\bshoelace\b',
        r'\bcap\b', r'\balarm\b', r'\bgym\b', r'\bmaintenance\b'
    ]

    ACCESSIBILITY_INDICATORS = {
        'jargon_terms': [
            'autophagy', 'oxidative stress', 'free radicals', 'antioxidants',
            'telomeres', 'inflammation', 'mitochondria', 'cellular', 'metabolism'
        ],
        'simple_language': [
            r'\byou\b', r'\byour\b', r'\bwhat\b', r'\bwhy\b', r'\bcan\b',
            r'\bhelp\b', r'\beasily\b', r'\bsimple\b'
        ]
    }

    DISCLAIMER_KEYWORDS = [
        r'\bconsult\b', r'\bdoctor\b', r'\bphysician\b', r'\bmedical professional\b',
        r'\bhealthcare\b', r'\bprofessional\b', r'\bnot medical advice\b',
        r'\bdiscuss\b', r'\bseek advice\b', r'\bspeak with\b', r'\bwarning\b',
        r'\bimportant\b', r'\balways\b', r'\bconcerns\b', r'\badvisable\b'
    ]

    ACTIONABLE_KEYWORDS = [
        r'\b\d+\.\s', r'\byou can\b', r'\btry to\b', r'\bconsider\b',
        r'\bfocus on\b', r'\baim for\b', r'\bsteps\b', r'\bstart\b',
        r'\bdo\b', r'\bpractice\b', r'\bmaintain\b', r'\binclude\b'
    ]

    MEDICAL_ACCURACY_POSITIVE = [
        r'\bscientific\b', r'\bresearch\b', r'\bstudies\b', r'\bevidence\b',
        r'\bproven\b', r'\bshown\b', r'\bhelps\b', r'\bsupports\b',
        r'\bprotects?\b', r'\bimproves?\b'
    ]

    def __init__(self):
        """Initialize the scoring framework."""
        self.responses = {}
        self.scores = {}
        self.detailed_scores = {}

    def add_response(self, response_id: str, text: str, topic: str):
        """Add a response to be scored."""
        self.responses[response_id] = {
            'text': text,
            'topic': topic
        }

    def count_matches(self, text: str, patterns: List[str]) -> int:
        """Count pattern matches in text."""
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate readability score based on:
        - Average sentence length
        - Average word length (via syllable count)
        - Returns score 1-5
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 3.0

        # Average sentence length (optimal is 10-20 words)
        words = text.split()
        avg_sentence_length = len(words) / len(sentences)

        # Calculate average syllables per word
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            syllables = 0
            vowels = "aeiou"
            previous_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllables += 1
                previous_was_vowel = is_vowel
            if word.endswith("e"):
                syllables -= 1
            if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
                syllables += 1
            return max(1, syllables)

        avg_syllables = np.mean([count_syllables(word) for word in words])

        # Score based on Flesch Kincaid metrics
        # Lower complexity is better (more accessible)
        complexity = avg_syllables

        if complexity <= 1.5 and 10 <= avg_sentence_length <= 20:
            return 5.0
        elif complexity <= 1.7 and 8 <= avg_sentence_length <= 22:
            return 4.0
        elif complexity <= 2.0 and 6 <= avg_sentence_length <= 25:
            return 3.0
        elif complexity <= 2.5:
            return 2.0
        else:
            return 1.0

    def score_medical_accuracy(self, text: str, topic: str) -> Tuple[float, Dict]:
        """
        Score medical accuracy based on:
        - Presence of scientific language
        - Absence of dangerous claims
        - Topic-specific accuracy indicators
        """
        details = {
            'scientific_claims': 0,
            'accurate_indicators': 0,
            'dangerous_keywords': 0,
            'topic_coverage': False
        }

        # Count scientific language
        details['scientific_claims'] = self.count_matches(text, self.MEDICAL_ACCURACY_POSITIVE)

        # Count accurate medical indicators
        details['accurate_indicators'] = self.count_matches(text, [
            r'\bcells\b', r'\bhealth\b', r'\bfunctions?\b', r'\bprotein\b',
            r'\bdisease\b', r'\bageing?\b'
        ])

        # Check for dangerous keywords (if absent, that's good)
        dangerous = [
            r'\bcure\b', r'\bguarantee\b', r'\bmagic\b', r'\bpanacea\b'
        ]
        details['dangerous_keywords'] = self.count_matches(text, dangerous)

        # Topic-specific coverage
        topic_keywords = {
            'Cellular Repair/Autophagy': [r'\bautophagy\b', r'\bcells?\b', r'\brenewal\b'],
            'Oxidative Stress': [r'\bfree radicals?\b', r'\bantioxidants?\b'],
            'Lifestyle & Cell Health': [r'\bsleep\b', r'\bexercise\b', r'\bfasting\b'],
            'Chronic Inflammation': [r'\binflammation\b', r'\bdamage\b'],
            'Telomeres': [r'\btelomeres?\b', r'\baging\b']
        }

        if topic in topic_keywords:
            topic_matches = self.count_matches(text, topic_keywords[topic])
            details['topic_coverage'] = topic_matches > 0

        # Calculate score (1-5)
        base_score = 3.0

        # Positive factors
        if details['scientific_claims'] > 0:
            base_score += 0.5
        if details['accurate_indicators'] >= 3:
            base_score += 0.5
        if details['topic_coverage']:
            base_score += 0.5

        # Negative factors
        if details['dangerous_keywords'] > 0:
            base_score -= 1.5

        score = max(1.0, min(5.0, base_score))
        return score, details

    def score_accessibility(self, text: str) -> Tuple[float, Dict]:
        """
        Score patient accessibility based on:
        - Readability
        - Jargon with explanations
        - Use of simple, direct language
        """
        details = {
            'readability_score': 0.0,
            'jargon_count': 0,
            'simple_language_count': 0,
            'explanation_present': False
        }

        # Calculate readability
        details['readability_score'] = self.calculate_readability_score(text)

        # Count jargon terms
        details['jargon_count'] = self.count_matches(text, [
            term for term in self.ACCESSIBILITY_INDICATORS['jargon_terms']
        ])

        # Count simple language
        details['simple_language_count'] = self.count_matches(
            text, self.ACCESSIBILITY_INDICATORS['simple_language']
        )

        # Check if jargon is explained (followed by "like", "is when", etc.)
        explanation_patterns = [
            r'\blike\b.*?\b(your|cells?|body)\b',
            r'\bis\s+(?:when|where|a)',
            r'\bthink of\b'
        ]
        details['explanation_present'] = self.count_matches(text, explanation_patterns) > 0

        # Calculate score
        base_score = details['readability_score']

        # Bonus for explaining jargon
        if details['jargon_count'] > 0 and details['explanation_present']:
            base_score += 0.5

        # Bonus for high simple language usage
        if details['simple_language_count'] >= 5:
            base_score += 0.5

        score = max(1.0, min(5.0, base_score))
        return score, details

    def score_analogy_quality(self, text: str) -> Tuple[float, Dict]:
        """
        Score analogy quality based on:
        - Number of analogies used
        - Clarity and relevance
        - Diversity of analogies
        """
        details = {
            'analogy_count': 0,
            'unique_analogies': [],
            'clarity_indicators': 0
        }

        # Find all analogies
        details['analogy_count'] = self.count_matches(text, self.ANALOGY_INDICATORS)

        # Extract actual analogies (look for patterns like "like X")
        analogy_patterns = [
            r"like\s+(?:your\s+)?(\w+(?:\s+\w+)?)",
            r"think\s+of\s+(?:your\s+)?(\w+(?:\s+\w+)?)",
            r"imagine\s+(?:your\s+)?(\w+(?:\s+\w+)?)"
        ]

        analogies = []
        for pattern in analogy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            analogies.extend(matches)

        details['unique_analogies'] = list(set(analogies))[:5]  # Top 5

        # Check for clarity indicators (e.g., "in other words", "essentially")
        clarity_patterns = [
            r'\bin other words\b', r'\bessentially\b', r'\bbasically\b',
            r'\bin simple terms\b', r'\bput simply\b'
        ]
        details['clarity_indicators'] = self.count_matches(text, clarity_patterns)

        # Calculate score
        base_score = 3.0

        if details['analogy_count'] == 0:
            return 1.0, details

        if details['analogy_count'] >= 3:
            base_score += 1.0
        elif details['analogy_count'] >= 2:
            base_score += 0.5

        if len(details['unique_analogies']) >= 2:
            base_score += 0.5

        if details['clarity_indicators'] > 0:
            base_score += 0.5

        score = max(1.0, min(5.0, base_score))
        return score, details

    def score_actionability(self, text: str) -> Tuple[float, Dict]:
        """
        Score actionability based on:
        - Numbered lists or bullet points
        - Action verbs and specific tips
        - Measurable goals
        """
        details = {
            'numbered_items': 0,
            'bullet_points': 0,
            'action_phrases': 0,
            'specific_metrics': 0,
            'action_list_present': False
        }

        # Count numbered items (1. 2. 3. etc.)
        details['numbered_items'] = len(re.findall(r'\d+\.\s+', text))

        # Count bullet points
        details['bullet_points'] = len(re.findall(r'[•\-\*]\s+', text))

        # Count action phrases
        details['action_phrases'] = self.count_matches(text, self.ACTIONABLE_KEYWORDS)

        # Find specific metrics (numbers with units)
        metrics = re.findall(r'\d+[-–]?\d*\s*(hours|minutes|days|weeks|grams|lbs|kg)', text, re.IGNORECASE)
        details['specific_metrics'] = len(metrics)

        details['action_list_present'] = details['numbered_items'] > 0 or details['bullet_points'] > 0

        # Calculate score
        base_score = 2.0

        if details['action_list_present']:
            base_score += 1.0

        if details['action_phrases'] >= 3:
            base_score += 1.0
        elif details['action_phrases'] >= 1:
            base_score += 0.5

        if details['specific_metrics'] > 0:
            base_score += 0.5

        score = max(1.0, min(5.0, base_score))
        return score, details

    def score_safety_disclaimers(self, text: str) -> Tuple[float, Dict]:
        """
        Score safety and disclaimer quality based on:
        - Presence of medical disclaimers
        - Strength of language
        - Professional advice recommendations
        """
        details = {
            'disclaimer_count': 0,
            'doctor_mention': 0,
            'professional_recommendation': 0,
            'warning_strength': 'none'
        }

        # Count disclaimer keywords
        details['disclaimer_count'] = self.count_matches(text, self.DISCLAIMER_KEYWORDS)

        # Specifically check for doctor mentions
        details['doctor_mention'] = self.count_matches(text, [
            r'\bdoctor\b', r'\bphysician\b', r'\bhealthcare\b'
        ])

        # Check for strong recommendations to see a professional
        professional_patterns = [
            r'(?:consult|discuss|speak with|see)\s+(?:your\s+)?(?:doctor|physician|healthcare|professional)',
            r'seek\s+medical\s+advice'
        ]
        details['professional_recommendation'] = self.count_matches(text, professional_patterns)

        # Determine warning strength
        if details['professional_recommendation'] > 0:
            details['warning_strength'] = 'strong'
        elif details['disclaimer_count'] >= 2:
            details['warning_strength'] = 'moderate'
        elif details['disclaimer_count'] >= 1:
            details['warning_strength'] = 'basic'

        # Calculate score
        score = 1.0

        if details['disclaimer_count'] >= 2:
            score += 1.0
        elif details['disclaimer_count'] >= 1:
            score += 0.5

        if details['doctor_mention'] > 0:
            score += 1.0

        if details['professional_recommendation'] > 0:
            score += 1.5

        score = max(1.0, min(5.0, score))
        return score, details

    def score_completeness(self, text: str, topic: str) -> Tuple[float, Dict]:
        """
        Score topic completeness based on:
        - Length and depth
        - Covering multiple aspects
        - Topic-specific requirements
        """
        details = {
            'word_count': 0,
            'aspect_coverage': 0,
            'depth_indicators': 0,
            'topic_specific_score': 0
        }

        words = text.split()
        details['word_count'] = len(words)

        # Define aspects to cover for each topic
        topic_aspects = {
            'Cellular Repair/Autophagy': [
                r'\bwhat\b', r'\bwhy\b', r'\bhow\b', r'\bbenefits?\b',
                r'\bprevent\b', r'\bsupport\b'
            ],
            'Oxidative Stress': [
                r'\bfree radicals?\b', r'\bantioxidants?\b', r'\bbalance\b',
                r'\bdamage\b', r'\bprotect\b'
            ],
            'Lifestyle & Cell Health': [
                r'\bsleep\b', r'\bexercise\b', r'\bfasting\b', r'\bdiet\b',
                r'\bstress\b'
            ],
            'Chronic Inflammation': [
                r'\binflamm\w*\b', r'\bacute\b', r'\bchronic\b', r'\bdamage\b',
                r'\bcauses?\b'
            ],
            'Telomeres': [
                r'\btelomeres?\b', r'\bshorten\b', r'\baging\b', r'\blifestyle\b',
                r'\bprotect\b'
            ]
        }

        if topic in topic_aspects:
            details['aspect_coverage'] = self.count_matches(text, topic_aspects[topic])

        # Count depth indicators (explanations, mechanisms)
        depth_patterns = [
            r'\bbecause\b', r'\bcauses?\b', r'\bleads? to\b', r'\bresults? in\b',
            r'\bmechanism\b', r'\bprocess\b', r'\bfunction\b'
        ]
        details['depth_indicators'] = self.count_matches(text, depth_patterns)

        # Calculate score
        score = 2.0

        if details['word_count'] >= 200:
            score += 1.0
        elif details['word_count'] >= 100:
            score += 0.5

        if details['aspect_coverage'] >= 3:
            score += 1.0
        elif details['aspect_coverage'] >= 2:
            score += 0.5

        if details['depth_indicators'] >= 2:
            score += 1.0
        elif details['depth_indicators'] >= 1:
            score += 0.5

        score = max(1.0, min(5.0, score))
        return score, details

    def score_response(self, response_id: str) -> Dict:
        """Score a single response on all criteria."""
        if response_id not in self.responses:
            raise ValueError(f"Response {response_id} not found")

        response = self.responses[response_id]
        text = response['text']
        topic = response['topic']

        scores = {
            'Medical Accuracy': self.score_medical_accuracy(text, topic),
            'Patient Accessibility': self.score_accessibility(text),
            'Analogy Quality': self.score_analogy_quality(text),
            'Actionability': self.score_actionability(text),
            'Safety/Disclaimers': self.score_safety_disclaimers(text),
            'Completeness': self.score_completeness(text, topic)
        }

        self.detailed_scores[response_id] = {
            criterion: {
                'score': score_tuple[0],
                'details': score_tuple[1]
            }
            for criterion, score_tuple in scores.items()
        }

        self.scores[response_id] = {
            criterion: score_tuple[0]
            for criterion, score_tuple in scores.items()
        }

        return self.detailed_scores[response_id]

    def score_all(self):
        """Score all responses."""
        for response_id in self.responses:
            self.score_response(response_id)

    def get_summary_table(self) -> str:
        """Generate a formatted summary table."""
        if not self.scores:
            return "No scores available. Run score_all() first."

        # Create header
        table = f"\n{Colors.BOLD}{Colors.CYAN}"
        table += "=" * 110 + "\n"
        table += "MEDGEMMA RESPONSE SCORING FRAMEWORK - SUMMARY RESULTS".center(110) + "\n"
        table += "=" * 110 + "\n"
        table += Colors.ENDC

        # Column headers
        headers = ['Response', 'Medical Acc.', 'Patient Access.', 'Analogy', 'Actionability', 'Safety', 'Completeness', 'AVERAGE']
        col_width = 12

        table += f"{Colors.BOLD}"
        table += f"{'Response':<15}"
        for header in headers[1:]:
            table += f"{header:>{col_width}}"
        table += f"{Colors.ENDC}\n"
        table += "-" * 110 + "\n"

        # Data rows
        all_averages = []
        for response_id in sorted(self.scores.keys()):
            scores_dict = self.scores[response_id]
            avg = np.mean(list(scores_dict.values()))
            all_averages.append(avg)

            # Color code based on score
            def color_score(score):
                if score >= 4.5:
                    return f"{Colors.GREEN}{score:.2f}{Colors.ENDC}"
                elif score >= 3.5:
                    return f"{Colors.CYAN}{score:.2f}{Colors.ENDC}"
                elif score >= 2.5:
                    return f"{Colors.YELLOW}{score:.2f}{Colors.ENDC}"
                else:
                    return f"{Colors.RED}{score:.2f}{Colors.ENDC}"

            table += f"{response_id:<15}"
            for criterion in ['Medical Accuracy', 'Patient Accessibility', 'Analogy Quality',
                            'Actionability', 'Safety/Disclaimers', 'Completeness']:
                score = scores_dict[criterion]
                table += f"{color_score(score):>{col_width}}"
            table += f"{color_score(avg):>{col_width}}\n"

        # Overall average
        table += "-" * 110 + "\n"
        overall_avg = np.mean(all_averages)
        table += f"{Colors.BOLD}OVERALL AVERAGE{Colors.ENDC:<20}"
        table += f"{color_score(overall_avg):>{col_width}}" * 8 + "\n"
        table += "=" * 110 + "\n\n"

        return table

    def create_radar_chart(self, output_path: str = 'scoring_radar.png'):
        """Create a radar chart of the scores."""
        if not self.scores:
            print("No scores available for chart")
            return

        categories = ['Medical\nAccuracy', 'Patient\nAccessibility', 'Analogy\nQuality',
                     'Actionability', 'Safety/\nDisclaimers', 'Completeness']

        # Setup the plot
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

        # Define colors for each response
        colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        # Plot data for each response
        for idx, response_id in enumerate(sorted(self.scores.keys())):
            scores_dict = self.scores[response_id]
            values = [scores_dict[cat] for cat in ['Medical Accuracy', 'Patient Accessibility',
                                                    'Analogy Quality', 'Actionability',
                                                    'Safety/Disclaimers', 'Completeness']]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=response_id, color=colors_list[idx])
            ax.fill(angles, values, alpha=0.15, color=colors_list[idx])

        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('MedGemma Response Quality Scores - Radar Chart\n', size=14, weight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{Colors.GREEN}Radar chart saved to: {output_path}{Colors.ENDC}")
        plt.close()

    def export_json(self, output_path: str = 'scoring_results.json'):
        """Export detailed scores to JSON."""
        export_data = {
            'metadata': {
                'framework': 'MedGemma Medical Accuracy Scoring Framework',
                'version': '1.0',
                'num_responses': len(self.scores)
            },
            'summary_scores': self.scores,
            'detailed_scores': {
                response_id: {
                    criterion: {
                        'score': round(details['score'], 2),
                        'details': details['details']
                    }
                    for criterion, details in score_dict.items()
                }
                for response_id, score_dict in self.detailed_scores.items()
            },
            'overall_metrics': {
                'average_score': round(np.mean([np.mean(list(scores.values()))
                                               for scores in self.scores.values()]), 2),
                'criteria_averages': {
                    criterion: round(np.mean([scores[criterion] for scores in self.scores.values()]), 2)
                    for criterion in ['Medical Accuracy', 'Patient Accessibility', 'Analogy Quality',
                                     'Actionability', 'Safety/Disclaimers', 'Completeness']
                }
            }
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"{Colors.GREEN}Results exported to: {output_path}{Colors.ENDC}")


def main():
    """Main execution function."""

    # Initialize framework
    framework = MedicalAccuracyScoringFramework()

    # Define the 5 MedGemma responses
    responses = {
        'Response 1': {
            'topic': 'Cellular Repair/Autophagy',
            'text': """Hi there! It's great you're asking about autophagy – it's a fascinating process that's like your body's internal recycling system. Think of your cells like tiny, busy factories. Over time, they accumulate 'junk' – old proteins, damaged components, and waste products that aren't needed anymore. This buildup can slow down their function and even lead to problems. Autophagy is the process where your cells clean house! It's like a cellular garbage truck that takes this 'junk' out, breaks it down into smaller pieces, and reuses those building blocks to create new, healthy components. This keeps your cells functioning optimally and helps protect you from diseases like cancer and neurodegenerative conditions. Why is this important for you? A healthy autophagy process means your cells are constantly being refreshed and repaired, which is fundamental to overall health, energy levels, and resilience against aging and disease. What can you do to support it? 1. Maintain a Healthy Weight 2. Eat a Balanced Diet: Focus on whole foods rich in antioxidants 3. Get Enough Sleep: Aim for 7-9 hours. Remember, this is just one piece of the puzzle for great health! Always discuss any concerns with your doctor."""
        },
        'Response 2': {
            'topic': 'Oxidative Stress',
            'text': """Oxidative stress occurs when your body has too many free radicals – think of them like tiny sparks that bounce around and cause damage to your cells. It's similar to rust forming on metal; just as oxygen can degrade iron, free radicals can damage cellular components. Your body has a natural defense system with antioxidants, which are like cellular bodyguards that neutralize these harmful sparks before they cause damage. When you have a healthy balance between free radicals and antioxidants, your cells stay protected and function well. However, when free radicals outnumber antioxidants (oxidative stress), cellular damage accumulates, contributing to aging and various diseases. What can support your antioxidant defenses? 1. Eat colorful fruits and vegetables rich in vitamins C and E 2. Include berries, nuts, and seeds in your diet 3. Reduce stress and get regular exercise 4. Limit exposure to pollution and excessive sun exposure 5. Consider consulting with a healthcare professional about supplements. Please note this is general health information, not medical advice. Always consult your doctor before starting supplements."""
        },
        'Response 3': {
            'topic': 'Lifestyle & Cell Health',
            'text': """Your lifestyle choices have a profound impact on how well your cells function. Think of cellular health as a symphony – everything needs to work in harmony. Sleep is your cells' night shift for maintenance: during deep sleep, your body repairs damage, consolidates memories, and removes waste products. Without adequate sleep, this maintenance can't happen effectively. Fasting, or giving your digestive system a break, allows your cells to focus their energy on repair and renewal rather than processing food. Exercise is like a cellular gym – it strengthens your cells' energy factories (mitochondria) and improves their resilience. A balanced diet provides the building blocks your cells need to maintain and repair themselves. Chronic stress, conversely, triggers inflammation that damages cells over time. To support cell health: 1. Sleep: Aim for 7-9 hours nightly 2. Movement: Exercise at least 30 minutes daily 3. Eating patterns: Consider intermittent fasting (but discuss with your doctor first) 4. Nutrition: Eat whole, nutrient-dense foods 5. Stress management: Practice meditation or yoga. Remember to consult a healthcare professional before making significant lifestyle changes, especially if you have existing health conditions."""
        },
        'Response 4': {
            'topic': 'Chronic Inflammation',
            'text': """Inflammation is your body's fire alarm system – when you get injured or face infection, inflammation signals your immune system to send help. This acute inflammation is protective and necessary. However, chronic inflammation is like a fire alarm that keeps ringing even after the fire is out. Your immune system remains activated, continuously releasing inflammatory molecules that damage healthy tissues. This persistent damage contributes to aging, heart disease, diabetes, and many other conditions. What causes chronic inflammation? Lifestyle factors like poor diet, sedentary behavior, chronic stress, insufficient sleep, and environmental toxins all fuel this harmful fire. The good news is you can help extinguish it: 1. Eat anti-inflammatory foods: fatty fish, leafy greens, berries, olive oil 2. Move regularly: Exercise reduces inflammation markers 3. Manage stress: Chronic stress amplifies inflammation 4. Prioritize sleep: Poor sleep increases inflammatory markers 5. Reduce processed foods: They often contain pro-inflammatory ingredients. Please note that if you have chronic inflammation or related conditions, it's crucial to consult with your healthcare provider for proper diagnosis and treatment. This information is educational only."""
        },
        'Response 5': {
            'topic': 'Telomeres',
            'text': """Telomeres are like the plastic caps on shoelaces – they protect the ends of your DNA strands and keep your genetic information intact. Each time your cell divides, telomeres shorten slightly. After many divisions, telomeres become too short, and the cell can no longer divide. This is a key mechanism of aging at the cellular level. Think of telomeres as a cellular clock – the shorter they are, the fewer divisions your cells can undergo. While you cannot stop telomere shortening entirely, your lifestyle choices significantly influence how quickly they shorten. Chronic stress, poor diet, sedentary behavior, and sleep deprivation all accelerate telomere shortening. Conversely, healthy habits can help preserve them. What supports telomere health? 1. Regular exercise: Studies show it slows telomere shortening 2. Mediterranean diet: Rich in antioxidants and anti-inflammatory foods 3. Stress reduction: Meditation and mindfulness practices 4. Quality sleep: Aim for 7-9 hours nightly 5. Social connections: Positive relationships reduce stress. While these recommendations are evidence-based, individual results vary. Always discuss your health goals with your doctor, especially if you're concerned about aging or cellular health."""
        }
    }

    # Add responses to framework
    for response_id, response_data in responses.items():
        framework.add_response(response_id, response_data['text'], response_data['topic'])

    # Score all responses
    print(f"\n{Colors.BOLD}{Colors.BLUE}Scoring MedGemma Responses...{Colors.ENDC}\n")
    framework.score_all()

    # Print summary table
    print(framework.get_summary_table())

    # Print detailed scores for each response
    print(f"{Colors.BOLD}{Colors.CYAN}DETAILED SCORING BREAKDOWN{Colors.ENDC}\n")
    print("=" * 110 + "\n")

    for response_id in sorted(framework.detailed_scores.keys()):
        topic = framework.responses[response_id]['topic']
        print(f"{Colors.BOLD}{response_id}: {topic}{Colors.ENDC}")
        print("-" * 80)

        for criterion, score_data in framework.detailed_scores[response_id].items():
            score = score_data['score']
            details = score_data['details']

            # Color based on score
            if score >= 4.5:
                color = Colors.GREEN
            elif score >= 3.5:
                color = Colors.CYAN
            elif score >= 2.5:
                color = Colors.YELLOW
            else:
                color = Colors.RED

            print(f"  {criterion:<25} {color}{score:.2f}/5.0{Colors.ENDC}")

            # Print key details
            for key, value in list(details.items())[:3]:
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                print(f"    • {key}: {value}")

        print()

    # Create radar chart
    print(f"\n{Colors.BOLD}{Colors.BLUE}Generating Visualizations...{Colors.ENDC}\n")
    framework.create_radar_chart('scoring_radar.png')

    # Export to JSON
    framework.export_json('scoring_results.json')

    # Print final summary statistics
    print(f"\n{Colors.BOLD}{Colors.CYAN}OVERALL FRAMEWORK METRICS{Colors.ENDC}")
    print("=" * 110)

    all_scores = [np.mean(list(scores.values())) for scores in framework.scores.values()]

    print(f"Average Overall Score:     {Colors.GREEN}{np.mean(all_scores):.2f}/5.0{Colors.ENDC}")
    print(f"Highest Score:             {Colors.GREEN}{np.max(all_scores):.2f}/5.0{Colors.ENDC}")
    print(f"Lowest Score:              {Colors.RED}{np.min(all_scores):.2f}/5.0{Colors.ENDC}")
    print(f"Standard Deviation:        {np.std(all_scores):.2f}")

    print(f"\nCriteria Averages:")
    for criterion in ['Medical Accuracy', 'Patient Accessibility', 'Analogy Quality',
                     'Actionability', 'Safety/Disclaimers', 'Completeness']:
        avg = np.mean([framework.scores[rid][criterion] for rid in framework.scores])
        print(f"  {criterion:<30} {avg:.2f}/5.0")

    print("\n" + "=" * 110 + "\n")


if __name__ == '__main__':
    main()
