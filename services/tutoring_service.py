"""
LearnLM-inspired Tutoring Service for Chat-O-Llama.
Based on pedagogical principles: active learning, step-by-step guidance, and adaptive teaching.
"""

import time
import logging
import threading
import re
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.llm_factory import get_active_backend
from services.conversation_manager import ConversationManager
from config import get_config

logger = logging.getLogger(__name__)


class TutoringService:
    """
    Main tutoring service implementing LearnLM-inspired pedagogical approaches.
    Uses sequential and parallel chaining for efficient processing.
    """
    
    # Query classification prompts
    CLASSIFIER_PROMPT = """Classify this student query into EXACTLY ONE category from the list below:

HOMEWORK_HELP - Specific homework problems, math equations, assignments, calculations, word problems that need solving
GUIDANCE - General study strategies, learning techniques, productivity advice, how to study better
NEW_TOPIC - Requests to learn or explain a new concept, subject, topic or field from scratch
CLARIFICATION - Questions about something confusing or unclear from previous discussion
ASSESSMENT - Requests to be quizzed, tested, or evaluated on a specific topic

Example classifications:
"How do I solve 2x + 5 = 15?" → HOMEWORK_HELP
"What's the best way to study for finals?" → GUIDANCE
"Can you teach me about photosynthesis?" → NEW_TOPIC
"I'm confused about what you said about derivatives" → CLARIFICATION
"Test my knowledge of Spanish vocabulary" → ASSESSMENT

Query: "{query}"

Return ONLY ONE WORD (the category name) with no explanation:"""

    # Pedagogical response prompts inspired by LearnLM principles - separated by category
    
    # GUIDANCE prompt for study advice
    PEDAGOGY_PROMPT_GUIDANCE = """You are a STUDY ADVISOR tutor helping a student with general guidance: "{query}"

Respond ONLY as a study advisor following these specific guidelines:
- FOCUS ONLY ON STUDY ADVICE, not content teaching
- Ask what specific study challenges they face
- Suggest practical learning strategies
- Provide time management or organizational tips
- Keep responses brief and actionable
- DO NOT teach specific subject matter
- DO NOT mention "20 minutes discussions" or "let's discuss issues/challenges"
- DO NOT refer to yourself as an "AI assistant"
- DO NOT use generic educational phrases repeatedly

Your response as a study advisor:"""

    # NEW_TOPIC prompt for introducing new concepts
    PEDAGOGY_PROMPT_NEW_TOPIC = """You are a CONCEPT INTRODUCTION tutor. The student wants to learn about: "{query}"

Respond ONLY as a concept introduction tutor following these specific guidelines:
- FOCUS ONLY ON INTRODUCING THE NEW TOPIC, not study advice
- Begin with "Let's explore [topic]..."
- Ask what they already know about this topic first
- Present 2-3 key foundational concepts only
- Use simple analogies to explain difficult ideas
- DO NOT overwhelm with too much information at once
- DO NOT mention "20 minutes discussions" or "let's discuss issues/challenges"
- DO NOT refer to yourself as an "AI assistant"
- DO NOT use generic educational phrases repeatedly

Your response as a concept introduction tutor:"""

    # HOMEWORK_HELP prompt for problem solving
    PEDAGOGY_PROMPT_HOMEWORK_HELP = """You are a PROBLEM-SOLVING tutor. The student needs help with this specific homework: "{query}"

Respond ONLY as a problem-solving tutor following these specific guidelines:
- FOCUS ONLY ON THIS SPECIFIC HOMEWORK PROBLEM, nothing else
- Begin with "Looking at this problem..."
- First identify the type of problem and relevant concepts
- DO NOT solve it directly - use the Socratic method
- Ask what specific step they're stuck on
- Provide a small hint toward the next step only
- DO NOT mention "20 minutes discussions" or "let's discuss issues/challenges"
- DO NOT refer to yourself as an "AI assistant"
- DO NOT use generic educational phrases repeatedly

Your response as a problem-solving tutor:"""

    # CLARIFICATION prompt for explaining confusing concepts
    PEDAGOGY_PROMPT_CLARIFICATION = """You are a CONCEPT CLARIFICATION tutor. The student is confused about: "{query}"

Respond ONLY as a clarification tutor following these specific guidelines:
- FOCUS ONLY ON CLEARING UP CONFUSION, not introducing new topics
- Begin with "Let me clarify..."
- Identify the specific misconception if possible
- Explain the challenging concept in simpler terms
- Use a different approach than was likely used before
- Provide a concrete example that illustrates the concept
- DO NOT mention "20 minutes discussions" or "let's discuss issues/challenges"
- DO NOT refer to yourself as an "AI assistant"
- DO NOT use generic educational phrases repeatedly

Your response as a clarification tutor:"""

    # ASSESSMENT prompt for knowledge testing
    PEDAGOGY_PROMPT_ASSESSMENT = """You are a KNOWLEDGE ASSESSMENT tutor. The student wants to test their understanding of: "{query}"

Respond ONLY as an assessment tutor following these specific guidelines:
- FOCUS ONLY ON TESTING KNOWLEDGE, not teaching new content
- Begin with a specific, targeted assessment question
- Make your question moderately challenging but fair
- Choose a question that reveals conceptual understanding
- Have a specific learning objective for your assessment
- DO NOT include the answer in your question
- DO NOT mention "20 minutes discussions" or "let's discuss issues/challenges"
- DO NOT refer to yourself as an "AI assistant"
- DO NOT use generic educational phrases repeatedly

Your assessment question:"""

    # Follow-up question generators for active learning - separated by category
    FOLLOWUP_PROMPT_GUIDANCE = "Generate a question about implementing this advice."
    FOLLOWUP_PROMPT_NEW_TOPIC = "Ask them to explain it in their own words."
    FOLLOWUP_PROMPT_HOMEWORK_HELP = "Ask what they learned from this approach."
    FOLLOWUP_PROMPT_CLARIFICATION = "Ask if they can apply this to a similar case."
    FOLLOWUP_PROMPT_ASSESSMENT = "Ask them to explain their reasoning."

    def __init__(self):
        self.config = get_config()
        self.backend = get_active_backend()
        self.executor = ThreadPoolExecutor(max_workers=3)
        logger.info("TutoringService initialized")
        # Helper lock or settings if needed

    def process_student_query(
        self, 
        query: str, 
        conversation_id: int,
        model: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing student queries with pedagogical approach.
        Uses sequential and parallel chaining for efficient processing.
        """
        start_time = time.time()
        
        try:
            # Get conversation history if not provided
            if conversation_history is None:
                conversation_history = self._get_conversation_history(conversation_id)
            
            # Sequential Step 1: Classify the query
            category = self._classify_query(query, conversation_history, model)
            logger.info(f"Query classified as: {category}")
            
            # Parallel processing: Generate main response and follow-up simultaneously
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit parallel tasks
                main_response_future = executor.submit(
                    self._generate_pedagogical_response, 
                    category, query, conversation_history, model
                )
                followup_future = executor.submit(
                    self._generate_followup_question, 
                    category, query, conversation_history, model
                )
                
                # Collect results
                main_response = main_response_future.result()
                followup_question = followup_future.result()
            
            # Sequential Step 2: Combine response with follow-up
            final_response = self._combine_response_with_followup(
                main_response, followup_question, category
            )
            
            # Save interaction to conversation history
            self._save_tutoring_interaction(
                conversation_id, query, final_response, category
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'response': final_response,
                'category': category,
                'response_time_ms': processing_time,
                'tutoring_mode': True,
                'backend_type': 'tutoring_llamacpp',
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Error in tutoring service: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            return {
                'response': "I apologize, but I encountered an error. Let me help you learn step by step. What specific topic would you like to explore?",
                'category': 'GUIDANCE',
                'response_time_ms': processing_time,
                'tutoring_mode': True,
                'error': True
            }

    def _classify_query(
        self, 
        query: str, 
        history: List[Dict[str, Any]], 
        model: str
    ) -> str:
        """Classify student query using LLM."""
        try:
            # Do keyword-based classification first for common homework patterns
            lower_query = query.lower()
            if any(keyword in lower_query for keyword in ["solve", "problem", "equation", "math", "calculate", 
                                                        "question", "answer", "homework", "assignment", 
                                                        "how many", "find the", "what is"]):
                logger.info(f"Query matched homework keywords: {query}")
                return "HOMEWORK_HELP"
                
            prompt = self.CLASSIFIER_PROMPT.format(query=query)
            
            response = self.backend.generate_response(
                model=model,
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            category = response.get('response', 'GUIDANCE').strip().upper()
            logger.info(f"LLM classified query as: {category}")
            
            # Validate category
            valid_categories = ['GUIDANCE', 'NEW_TOPIC', 'HOMEWORK_HELP', 'CLARIFICATION', 'ASSESSMENT']
            if category not in valid_categories:
                category = 'GUIDANCE'  # Default fallback
                
            return category
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return 'GUIDANCE'  # Safe fallback
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Convert conversation history to a text block for prompt insertion."""
        if not history:
            return ''
        lines = []
        for msg in history:
            role = msg.get('role', 'user')
            # Strip category tag from assistant messages
            raw = msg.get('content', '').strip()
            if role == 'assistant':
                content = re.sub(r'^\[[A-Z_]+\]\s*', '', raw)
            else:
                content = raw
            if content:
                # Normalize role names
                role_label = 'Human' if role == 'user' else 'Assistant'
                lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def _generate_pedagogical_response(
        self, 
        category: str, 
        query: str, 
        history: List[Dict[str, Any]], 
        model: str
    ) -> str:
        """Generate pedagogical response based on category and LearnLM principles."""
        try:
            # Select the appropriate prompt based on category
            if category == "GUIDANCE":
                base_prompt = self.PEDAGOGY_PROMPT_GUIDANCE.format(query=query)
            elif category == "NEW_TOPIC":
                base_prompt = self.PEDAGOGY_PROMPT_NEW_TOPIC.format(query=query)
            elif category == "HOMEWORK_HELP":
                base_prompt = self.PEDAGOGY_PROMPT_HOMEWORK_HELP.format(query=query)
            elif category == "CLARIFICATION":
                base_prompt = self.PEDAGOGY_PROMPT_CLARIFICATION.format(query=query)
            elif category == "ASSESSMENT":
                base_prompt = self.PEDAGOGY_PROMPT_ASSESSMENT.format(query=query)
            else:
                # Fallback to guidance prompt if category is unknown
                base_prompt = self.PEDAGOGY_PROMPT_GUIDANCE.format(query=query)
                
            history_text = self._format_history(history)
            
            # Select the appropriate system message based on category
            if category == "GUIDANCE":
                system_message = """SYSTEM: You are a STUDY ADVISOR helping with learning strategies. 
- Focus exclusively on practical study advice and strategies
- Stay focused on HOW to study, not WHAT to study
- Be specific and actionable in your advice"""
                
            elif category == "NEW_TOPIC":
                system_message = """SYSTEM: You are a CONCEPT INTRODUCTION tutor introducing new topics.
- Focus on foundational concepts only
- Use intuitive explanations and clear analogies
- Build understanding from first principles"""
                
            elif category == "HOMEWORK_HELP":
                system_message = """SYSTEM: You are a PROBLEM-SOLVING tutor for homework.
- Focus exclusively on guiding through this specific problem
- Use leading questions to help them reach the answer themselves
- Do not solve the problem directly, use step-by-step guidance"""
                
            elif category == "CLARIFICATION":
                system_message = """SYSTEM: You are a CONCEPT CLARIFICATION tutor.
- Focus only on the specific confusion point
- Use alternative explanations and concrete examples
- Check understanding with targeted questions"""
                
            elif category == "ASSESSMENT":
                system_message = """SYSTEM: You are a KNOWLEDGE ASSESSMENT tutor.
- Focus on testing understanding with targeted questions
- Ask questions that reveal conceptual understanding
- Use precise, challenging but fair assessment questions"""
            
            else:
                # Default system message if category is unknown
                system_message = """SYSTEM: You are a helpful educational tutor. Respond directly to the student's question."""
            
            # Add common instructions to all system messages
            system_message += """
- DO NOT repeat phrases like "I want to guide them through a deep discussion"
- DO NOT mention "20 minutes discussions" or similar phrases
- DO NOT refer to yourself as an "AI assistant"
- AVOID repetitive patterns in your responses
- Be natural, conversational and to-the-point"""
            
            if history_text:
                prompt = f"{system_message}\n\nConversation history:\n{history_text}\n\n{base_prompt}"
            else:
                prompt = f"{system_message}\n\n{base_prompt}"
            
            response = self.backend.generate_response(
                model=model,
                prompt=prompt,
                max_tokens=250,
                temperature=0.7
            )
            
            return response.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error generating pedagogical response: {e}")
            return "Let me help you learn this step by step. What would you like to start with?"

    def _generate_followup_question(
        self, 
        category: str, 
        query: str, 
        history: List[Dict[str, Any]], 
        model: str
    ) -> str:
        """Generate follow-up question to promote active learning."""
        try:
            # Select the appropriate follow-up prompt based on category
            if category == "GUIDANCE":
                followup_prompt = self.FOLLOWUP_PROMPT_GUIDANCE
            elif category == "NEW_TOPIC":
                followup_prompt = self.FOLLOWUP_PROMPT_NEW_TOPIC
            elif category == "HOMEWORK_HELP":
                followup_prompt = self.FOLLOWUP_PROMPT_HOMEWORK_HELP
            elif category == "CLARIFICATION":
                followup_prompt = self.FOLLOWUP_PROMPT_CLARIFICATION
            elif category == "ASSESSMENT":
                followup_prompt = self.FOLLOWUP_PROMPT_ASSESSMENT
            else:
                # Fallback to guidance prompt if category is unknown
                followup_prompt = self.FOLLOWUP_PROMPT_GUIDANCE
            
            prompt = f"""Student asked: "{query}"

As a tutor, generate ONE short follow-up question.
{followup_prompt}

Rules:
- Keep it under 15 words
- Make it direct and specific
- Don't mention "AI assistant" or "20 minute discussions"
- No explanations, just the question

Your question:"""
            
            response = self.backend.generate_response(
                model=model,
                prompt=prompt,
                max_tokens=30,
                temperature=0.7
            )
            
            return response.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error generating follow-up question: {e}")
            return "What questions do you have about what we just discussed?"

    def _combine_response_with_followup(
        self, 
        main_response: str, 
        followup_question: str, 
        category: str
    ) -> str:
        """Combine main pedagogical response with follow-up question."""
        if not followup_question:
            return main_response
        
        # Clean up the followup question if it starts with redundant text
        if followup_question.lower().startswith(('question:', 'follow-up:', 'ask:')):
            followup_question = followup_question.split(':', 1)[1].strip()
        
        # Combine with appropriate spacing and formatting
        combined = f"{main_response}\n\n💡 {followup_question}"
        
        return combined

    def _get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get conversation history for context."""
        try:
            messages = ConversationManager.get_messages(conversation_id)
            history = []
            
            for msg in messages[-10:]:  # Last 10 messages for context
                history.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def _format_history_for_prompt(
        self, 
        history: List[Dict[str, Any]], 
        limit: int = 5
    ) -> str:
        """Unused helper - no operation"""
        pass

    def _save_tutoring_interaction(
        self, 
        conversation_id: int, 
        query: str, 
        response: str, 
        category: str
    ):
        """Save the tutoring interaction to conversation history."""
        try:
            # Add user message
            ConversationManager.add_message(
                conversation_id=conversation_id,
                role='user',
                content=query
            )
            
            # Add tutor response with category metadata
            tutor_response = f"[{category}] {response}"
            ConversationManager.add_message(
                conversation_id=conversation_id,
                role='assistant',
                content=tutor_response
            )
            
            # Update conversation timestamp
            ConversationManager.update_conversation_timestamp(conversation_id)
            
        except Exception as e:
            logger.error(f"Error saving tutoring interaction: {e}")

    def get_tutoring_analytics(self, conversation_id: int) -> Dict[str, Any]:
        """Get analytics about the tutoring session."""
        try:
            messages = ConversationManager.get_messages(conversation_id)
            
            # Analyze message categories
            categories = {'GUIDANCE': 0, 'NEW_TOPIC': 0, 'HOMEWORK_HELP': 0, 
                         'CLARIFICATION': 0, 'ASSESSMENT': 0}
            
            tutor_messages = 0
            student_messages = 0
            
            for msg in messages:
                if msg['role'] == 'user':
                    student_messages += 1
                elif msg['role'] == 'assistant' and msg['content'].startswith('['):
                    tutor_messages += 1
                    # Extract category from response
                    try:
                        category = msg['content'].split(']')[0][1:]
                        if category in categories:
                            categories[category] += 1
                    except:
                        pass
            
            return {
                'total_interactions': len(messages) // 2,
                'student_messages': student_messages,
                'tutor_responses': tutor_messages,
                'category_breakdown': categories,
                'most_common_category': max(categories, key=categories.get),
                'engagement_ratio': student_messages / max(tutor_messages, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting tutoring analytics: {e}")
            return {}

    def cleanup(self):
        """Clean up resources."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")