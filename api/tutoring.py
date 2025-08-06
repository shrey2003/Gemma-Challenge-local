"""Tutoring API routes for LearnLM-inspired tutoring mode."""

import logging
from flask import request, jsonify, Blueprint
from services.tutoring_service import TutoringService
from services.conversation_manager import ConversationManager
from services.llm_factory import get_llm_factory
from config import get_config

logger = logging.getLogger(__name__)

tutoring_bp = Blueprint('tutoring', __name__)


@tutoring_bp.route('/api/tutoring/test', methods=['GET'])
def tutoring_test():
    """Simple test endpoint to verify tutoring blueprint is working."""
    return jsonify({
        'success': True,
        'message': 'Tutoring API is working',
        'endpoint': 'test'
    })

# Initialize tutoring service (lazy loading to avoid startup errors)
tutoring_service = None

def get_tutoring_service():
    global tutoring_service
    if tutoring_service is None:
        tutoring_service = TutoringService()
    return tutoring_service


@tutoring_bp.route('/api/tutoring/chat', methods=['POST'])
def tutoring_chat():
    """
    Handle tutoring chat requests with pedagogical approach.
    Uses sequential and parallel chaining for efficient processing.
    """
    logger.info("Tutoring chat endpoint called")
    try:
        data = request.get_json()
        logger.info(f"Received tutoring chat data: {data}")
        
        # Validate required fields
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message is required',
                'success': False
            }), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({
                'error': 'Message cannot be empty',
                'success': False
            }), 400
        
        # Get conversation details
        conversation_id = data.get('conversation_id')
        model = data.get('model')
        
        # Validate conversation exists
        if conversation_id:
            conversation = ConversationManager.get_conversation(conversation_id)
            if not conversation:
                return jsonify({
                    'error': 'Conversation not found',
                    'success': False
                }), 404
        else:
            # Create new tutoring conversation
            factory = get_llm_factory()
            available_models = factory.get_available_models()
            
            # Use first available model if none specified
            if not model:
                for backend_type, models in available_models.items():
                    if models:
                        model = models[0]
                        break
                
                if not model:
                    return jsonify({
                        'error': 'No models available',
                        'success': False
                    }), 503
            
            # Create new conversation for tutoring (use regular backend)
            conversation_id = ConversationManager.create_conversation(
                title=f"Tutoring: {message[:30]}...",
                model=model,
                backend_type='llamacpp'  # Use regular backend, not tutoring_llamacpp
            )
        
        # Process the student query with tutoring service
        service = get_tutoring_service()
        result = service.process_student_query(
            query=message,
            conversation_id=conversation_id,
            model=model
        )
        
        # Return tutoring response
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'response': result['response'],
            'category': result['category'],
            'response_time_ms': result['response_time_ms'],
            'tutoring_mode': True,
            'model': model,
            'backend_type': result.get('backend_type', 'tutoring_llamacpp')
        })
        
    except Exception as e:
        logger.error(f"Error in tutoring chat: {e}")
        return jsonify({
            'error': f'Tutoring error: {str(e)}',
            'success': False
        }), 500


@tutoring_bp.route('/api/tutoring/conversations', methods=['GET'])
def get_tutoring_conversations():
    """Get all tutoring conversations."""
    try:
        conversations = ConversationManager.get_conversations()
        
        # Filter for tutoring conversations (those with "Tutoring:" in title)
        tutoring_conversations = [
            dict(conv) for conv in conversations 
            if conv['title'] and conv['title'].startswith('Tutoring:')
        ]
        
        return jsonify({
            'success': True,
            'conversations': tutoring_conversations
        })
        
    except Exception as e:
        logger.error(f"Error getting tutoring conversations: {e}")
        return jsonify({
            'error': f'Error retrieving conversations: {str(e)}',
            'success': False
        }), 500


@tutoring_bp.route('/api/tutoring/conversations/<int:conversation_id>', methods=['GET'])
def get_tutoring_conversation(conversation_id):
    """Get a specific tutoring conversation with messages."""
    try:
        # Get conversation details
        conversation = ConversationManager.get_conversation(conversation_id)
        if not conversation:
            return jsonify({
                'error': 'Conversation not found',
                'success': False
            }), 404
        
        # Verify it's a tutoring conversation
        if not (conversation['title'] and conversation['title'].startswith('Tutoring:')):
            return jsonify({
                'error': 'Not a tutoring conversation',
                'success': False
            }), 400
        
        # Get messages
        messages = ConversationManager.get_messages(conversation_id)
        
        return jsonify({
            'success': True,
            'conversation': dict(conversation),
            'messages': [dict(msg) for msg in messages]
        })
        
    except Exception as e:
        logger.error(f"Error getting tutoring conversation: {e}")
        return jsonify({
            'error': f'Error retrieving conversation: {str(e)}',
            'success': False
        }), 500


@tutoring_bp.route('/api/tutoring/conversations/<int:conversation_id>/analytics', methods=['GET'])
def get_tutoring_analytics(conversation_id):
    """Get tutoring analytics for a conversation."""
    try:
        # Verify conversation exists and is tutoring conversation
        conversation = ConversationManager.get_conversation(conversation_id)
        if not conversation:
            return jsonify({
                'error': 'Conversation not found',
                'success': False
            }), 404
        
        if not (conversation['title'] and conversation['title'].startswith('Tutoring:')):
            return jsonify({
                'error': 'Not a tutoring conversation',
                'success': False
            }), 400
        
        # Get analytics from tutoring service
        service = get_tutoring_service()
        analytics = service.get_tutoring_analytics(conversation_id)
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'analytics': analytics
        })
        
    except Exception as e:
        logger.error(f"Error getting tutoring analytics: {e}")
        return jsonify({
            'error': f'Error retrieving analytics: {str(e)}',
            'success': False
        }), 500


@tutoring_bp.route('/api/tutoring/categories', methods=['GET'])
def get_tutoring_categories():
    """Get available tutoring categories and their descriptions."""
    try:
        categories = {
            'GUIDANCE': {
                'name': 'Study Guidance',
                'description': 'General study advice, learning strategies, and academic guidance',
                'icon': '🎯',
                'approach': 'Reflective questioning and strategy discovery'
            },
            'NEW_TOPIC': {
                'name': 'Learn New Topic',
                'description': 'Learning new concepts, subjects, or topics from scratch',
                'icon': '📚',
                'approach': 'Step-by-step discovery learning with active engagement'
            },
            'HOMEWORK_HELP': {
                'name': 'Homework Help',
                'description': 'Assistance with specific homework problems and assignments',
                'icon': '📝',
                'approach': 'Guided problem-solving without direct answers'
            },
            'CLARIFICATION': {
                'name': 'Clarification',
                'description': 'Questions about previously discussed topics or follow-up questions',
                'icon': '❓',
                'approach': 'Targeted explanation with multiple perspectives'
            },
            'ASSESSMENT': {
                'name': 'Assessment',
                'description': 'Testing knowledge and evaluating understanding',
                'icon': '🎓',
                'approach': 'Progressive questioning with constructive feedback'
            }
        }
        
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        logger.error(f"Error getting tutoring categories: {e}")
        return jsonify({
            'error': f'Error retrieving categories: {str(e)}',
            'success': False
        }), 500


@tutoring_bp.route('/api/tutoring/switch', methods=['POST'])
def switch_to_tutoring():
    """Switch an existing conversation to tutoring mode."""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        
        if not conversation_id:
            return jsonify({
                'error': 'Conversation ID is required',
                'success': False
            }), 400
        
        # Get conversation
        conversation = ConversationManager.get_conversation(conversation_id)
        if not conversation:
            return jsonify({
                'error': 'Conversation not found',
                'success': False
            }), 404
        
        # Update conversation title to indicate tutoring mode
        # We'll modify the title to start with "Tutoring:" to mark it as a tutoring conversation
        from utils.database import get_db
        db = get_db()
        current_title = conversation['title']
        if not current_title.startswith('Tutoring:'):
            new_title = f"Tutoring: {current_title}"
            db.execute(
                'UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (new_title, conversation_id)
            )
            db.commit()
        
        return jsonify({
            'success': True,
            'message': 'Conversation switched to tutoring mode',
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error switching to tutoring mode: {e}")
        return jsonify({
            'error': f'Error switching to tutoring mode: {str(e)}',
            'success': False
        }), 500


@tutoring_bp.route('/api/tutoring/health', methods=['GET'])
def tutoring_health_check():
    """Health check for tutoring service."""
    try:
        # Check if tutoring service is available
        factory = get_llm_factory()
        available_models = factory.get_available_models()
        
        has_models = any(models for models in available_models.values())
        
        status = {
            'success': True,
            'tutoring_service': 'available',
            'models_available': has_models,
            'available_models': available_models,
            'features': [
                'Query Classification',
                'Pedagogical Response Generation',
                'Sequential and Parallel Chaining',
                'Active Learning Promotion',
                'Conversation Analytics'
            ]
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in tutoring health check: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'tutoring_service': 'unavailable'
        }), 500