from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import traceback

from config import Config
from models.face_recognition import FaceRecognitionModel
from database import (
    store_face_embedding, 
    store_license_record, 
    user_exists,
    get_license_record,
    get_all_users,
    delete_user_data
)

registration_bp = Blueprint('registration', __name__)

# Initialize model
face_model = FaceRecognitionModel()

@registration_bp.route('/register', methods=['POST'])
def register_user_endpoint():
    """
    Register a new user with face and license information
    """
    try:
        # Get form data
        user_id = request.form.get('user_id')
        name = request.form.get('name')
        license_number = request.form.get('license_number')
        dob = request.form.get('dob')
        address = request.form.get('address')
        face_image = request.files.get('face_image')
        
        # Validate inputs
        if not all([user_id, name, license_number, dob, address, face_image]):
            return jsonify({
                "success": False,
                "message": "All fields are required"
            }), 400
        
        print(f"=" * 60)
        print(f"REGISTRATION REQUEST")
        print(f"=" * 60)
        print(f"User ID: {user_id}")
        print(f"Name: {name}")
        print(f"License Number: {license_number}")
        print(f"DOB: {dob}")
        print(f"Address: {address}")
        
        # Check if user already exists
        if user_exists(user_id):
            return jsonify({
                "success": False,
                "message": f"User ID '{user_id}' already exists"
            }), 400
        
        # Save face image
        face_filename = secure_filename(f'{user_id}_face.jpg')
        face_path = os.path.join(Config.UPLOAD_FOLDER, face_filename)
        face_image.save(face_path)
        
        print(f"📁 Face image saved: {face_path}")
        
        # Extract face embedding
        print(f"🔄 Extracting face embedding...")
        embedding = face_model.extract_embedding(face_path)
        
        if embedding is None:
            os.remove(face_path)  # Clean up
            return jsonify({
                "success": False,
                "message": "No face detected in the uploaded image"
            }), 400
        
        print(f"✅ Face embedding extracted (dimension: {len(embedding)})")

        # Reject if this face is already registered under another user
        print("🔄 Checking for duplicate face...")
        duplicate_user, similarity = face_model.find_duplicate_face(embedding)
        if duplicate_user:
            os.remove(face_path)  # Clean up uploaded file
            print(f"❌ Duplicate face detected - matches user '{duplicate_user}' (similarity: {similarity:.4f})")
            return jsonify({
                "success": False,
                "message": f"This face is already registered under user ID '{duplicate_user}'. Duplicate registrations are not allowed."
            }), 409

        # Store face embedding in database
        print("🔄 Storing face embedding in database...")
        store_face_embedding(user_id, embedding if isinstance(embedding, list) else embedding.tolist(), model_name='OpenCV-Custom')
        print(f"✅ Face embedding stored")
        
        # Store license information
        print("🔄 Storing license information...")
        license_data = {
            "name": name,
            "license_number": license_number,
            "dob": dob,
            "address": address
        }
        
        store_license_record(user_id, license_data)
        print(f"✅ License record stored")
        
        # Verify it was stored
        stored_license = get_license_record(user_id)
        license_stored = stored_license is not None
        
        if not license_stored:
            print("⚠️ WARNING: License record not found after storage!")
        
        print("=" * 60)
        print(f"✅ REGISTRATION SUCCESSFUL!")
        print(f"   Face embedding: Stored")
        print(f"   License record: {'Stored' if license_stored else 'FAILED'}")
        print("=" * 60)
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "details": {
                "face_registered": True,
                "license_stored": license_stored,
                "embedding_dimension": len(embedding)
            }
        }), 201
        
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Registration failed: {str(e)}"
        }), 500


@registration_bp.route('/users', methods=['GET'])
def list_users():
    """Get list of all registered users"""
    try:
        users = get_all_users()
        
        return jsonify({
            "success": True,
            "count": len(users),
            "users": users
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching users: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@registration_bp.route('/users/<user_id>', methods=['GET'])
def get_user_details(user_id):
    """Get details for specific user"""
    try:
        from database import get_face_embedding
        
        embedding = get_face_embedding(user_id)
        license_data = get_license_record(user_id)
        
        if not embedding:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "has_face": embedding is not None,
            "has_license": license_data is not None,
            "license_data": license_data
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching user: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@registration_bp.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user and their data"""
    try:
        if not user_exists(user_id):
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404
        
        delete_user_data(user_id)
        
        return jsonify({
            "success": True,
            "message": f"User {user_id} deleted successfully"
        }), 200
            
    except Exception as e:
        print(f"❌ Error deleting user: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
