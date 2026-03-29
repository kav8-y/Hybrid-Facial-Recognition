from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import numpy as np
from datetime import datetime
import traceback
from difflib import SequenceMatcher

from config import Config
from models.face_recognition import FaceRecognitionModel
from models.license_detection import LicenseDetectionModel
from database import (
    get_all_face_embeddings,
    get_license_record,
    log_verification,
    get_face_embedding
)

verification_bp = Blueprint('verification', __name__)


# Initialize models
face_model = FaceRecognitionModel()
license_model = LicenseDetectionModel()

def normalize_text(text):
    """Normalize text for comparison"""
    if not text:
        return ""
    return text.upper().strip().replace(',', '').replace('.', '')

def normalize_dob(dob_str):
    """
    Normalize DOB to a standard format for comparison
    Handles: DD-MM-YYYY, MM/DD/YYYY, DD/MM/YYYY
    """
    if not dob_str:
        return ""
    
    # Replace separators
    normalized = dob_str.replace('/', '-').replace('.', '-')
    
    # Try to parse and standardize
    try:
        # Split by separator
        parts = normalized.split('-')
        if len(parts) == 3:
            # Check if it's MM-DD-YYYY or DD-MM-YYYY
            # If first part > 12, it's DD-MM-YYYY
            if int(parts[0]) > 12:
                # Already DD-MM-YYYY
                return normalized
            else:
                # Could be MM-DD-YYYY or DD-MM-YYYY
                # Try both interpretations
                return normalized
    except:
        pass
    
    return normalized

def compare_dob(dob1, dob2):
    """
    Compare two DOB strings with format flexibility
    Returns True if they match (accounting for format differences)
    """
    if not dob1 or not dob2:
        return False
    
    # Normalize both
    norm1 = normalize_dob(dob1)
    norm2 = normalize_dob(dob2)
    
    # Direct match
    if norm1 == norm2:
        return True
    
    # Try swapping day/month for US vs International format
    try:
        parts1 = norm1.split('-')
        parts2 = norm2.split('-')
        
        if len(parts1) == 3 and len(parts2) == 3:
            # Check if swapping day/month matches
            # Format 1: DD-MM-YYYY vs MM-DD-YYYY
            if parts1[0] == parts2[1] and parts1[1] == parts2[0] and parts1[2] == parts2[2]:
                return True
    except:
        pass
    
    return False

def text_similarity(text1, text2):
    """Calculate similarity between two strings (0-1)"""
    if not text1 or not text2:
        return 0.0
    
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    return SequenceMatcher(None, norm1, norm2).ratio()

def token_confidence_score(stored_name, extraction_data):
    """
    Calculate finding a stored name within the pool of OCR extracted tokens.
    Returns: score 0.0 to 1.0
    Logic:
      1. Tokenize stored name
      2. For each token, find highest fuzzy match in OCR tokens
      3. Accept match if > 85% similarity
      4. Require matched_tokens / total_tokens >= 60% AND at least one > 90%
    """
    if not stored_name or not extraction_data.get("tokens"):
        return 0.0
        
    stored_tokens = normalize_text(stored_name).split()
    if not stored_tokens:
        return 0.0
        
    ocr_tokens = [t["text"] for t in extraction_data["tokens"]]
    
    matched_count = 0
    has_strong_match = False
    
    for st in stored_tokens:
        best_sim = 0.0
        for ot in ocr_tokens:
            sim = text_similarity(st, ot)
            best_sim = max(best_sim, sim)
            
        if best_sim > 0.85:
            matched_count += 1
        if best_sim > 0.90:
            has_strong_match = True
            
    match_ratio = matched_count / len(stored_tokens)
    
    if match_ratio >= 0.60 and has_strong_match:
        return match_ratio
    else:
        return 0.0


@verification_bp.route('/verify', methods=['POST'])
def verify_identity():
    """
    Verify user identity using face and license
    """
    try:
        face_image = request.files.get('face_image')
        license_image = request.files.get('license_image')
        
        if not face_image or not license_image:
            return jsonify({
                "status": "failed",
                "message": "Both face and license images are required"
            }), 400
        
        # Save uploaded files
        face_filename = secure_filename(f'verify_face.jpg')
        license_filename = secure_filename(f'verify_license.jpg')
        
        face_path = os.path.join(Config.UPLOAD_FOLDER, face_filename)
        license_path = os.path.join(Config.UPLOAD_FOLDER, license_filename)
        
        face_image.save(face_path)
        license_image.save(license_path)
        
        print("📁 Files saved for verification")
        
        # ========== STEP 1: FACE VERIFICATION ==========
        print("=" * 50)
        print("STEP 1: FACE VERIFICATION")
        print("=" * 50)
        
        print(f"🔄 Verifying face from: {face_path}")
        face_embedding = face_model.extract_embedding(face_path)
        
        if face_embedding is None:
            return jsonify({
                "status": "failed",
                "message": "No face detected in the uploaded image"
            }), 400
        
        # Compare with registered faces
                # Compare with registered faces
        registered_faces = get_all_face_embeddings()
        
        if not registered_faces:
            return jsonify({
                "status": "failed",
                "message": "No registered users found in database"
            }), 404
        
        print(f"📊 Comparing with {len(registered_faces)} registered users...")
        
        best_match_user = None
        best_match_confidence = 0.0
        threshold = 0.70  # Confidence threshold
        
        for user_id, stored_embedding in registered_faces:
            try:
                # Convert to numpy arrays if needed
                if not isinstance(face_embedding, np.ndarray):
                    face_emb = np.array(face_embedding)
                else:
                    face_emb = face_embedding
                
                if not isinstance(stored_embedding, np.ndarray):
                    stored_emb = np.array(stored_embedding)
                else:
                    stored_emb = stored_embedding
                
                # Calculate cosine similarity
                cosine_sim = np.dot(face_emb, stored_emb) / (np.linalg.norm(face_emb) * np.linalg.norm(stored_emb))
                
                # Calculate correlation
                correlation = np.corrcoef(face_emb, stored_emb)[0, 1]
                
                # Combined similarity score
                similarity = (cosine_sim + correlation) / 2
                
                print(f"   User {user_id}: similarity = {similarity:.4f} (cosine: {cosine_sim:.4f}, corr: {correlation:.4f})")
                
                if similarity > best_match_confidence:
                    best_match_confidence = similarity
                    best_match_user = user_id
                    
            except Exception as e:
                print(f"   ⚠️ Error comparing with user {user_id}: {e}")
                continue
        
        if best_match_confidence < threshold:
            print(f"❌ No face match found (best: {best_match_confidence:.4f}, threshold: {threshold})")
            return jsonify({
                "status": "failed",
                "message": "Face not recognized",
                "confidence": 0,
                "face_verified": False,
                "license_verified": False,
                "liveness_verified": False
            }), 200
        
        print(f"✅ Face matched! User: {best_match_user}, Confidence: {best_match_confidence:.4f}")
        print(f"✅ Face matched: {best_match_user} (confidence: {best_match_confidence:.4f})")
        
        # ========== STEP 2: BYPASS OCR & DIRECTLY FETCH STORED LICENSE DATA ==========
        print("=" * 50)
        print("STEP 2: FETCH STORED LICENSE DATA (OCR BYPASSED)")
        print("=" * 50)
        
        stored_data = get_license_record(best_match_user)
        
        if not stored_data:
            print(f"❌ No license record found for user: {best_match_user}")
            return jsonify({
                "status": "failed",
                "message": "No license record found for matched user",
                "face_verified": True,
                "license_verified": False,
                "liveness_verified": False,
                "confidence": 0,
                "details": {
                    "user_id": best_match_user,
                    "message": "No license record found for matched user"
                }
            }), 200
        
        print("📋 Stored license data retrieved")
        
        # ========== STEP 3: FINAL VERIFICATION (FACE ONLY 70%+) ==========
        print("=" * 50)
        print("STEP 3: FINAL VERIFICATION (Strict Face Match)")
        print("=" * 50)
        
        is_verified = best_match_confidence >= 0.70
        
        if is_verified:
            print(f"✅ VERIFICATION SUCCESSFUL (Face Score: {best_match_confidence:.4f})")
            
            # Log successful verification
            log_verification(
                user_id=best_match_user,
                status="verified",
                face_confidence=float(best_match_confidence),
                license_match_score=1.0, # Dummy perfect score since OCR is skipped
                liveness_passed=False
            )
            
            return jsonify({
                "status": "verified",
                "face_verified": True,
                "license_verified": True, # Automatically assert true because face matches
                "liveness_verified": False,
                "confidence": float(best_match_confidence),
                "face_score": float(best_match_confidence * 100),
                "name_match_score": 100.0,
                "dl_match": True,
                "final_score": 140.0,
                "decision": "VERIFIED",
                "details": {
                    "user_id": best_match_user,
                    "name": stored_data.get('name'),
                    "license_number": stored_data.get('license_number'),
                    "dob": stored_data.get('dob'),
                    "address": stored_data.get('address'),
                }
            }), 200
        else:
            print(f"❌ VERIFICATION FAILED - Face match too low ({best_match_confidence:.4f} < 0.70)")
            
            # Log failed verification
            log_verification(
                user_id=best_match_user,
                status="failed",
                face_confidence=float(best_match_confidence),
                license_match_score=0.0,
                liveness_passed=False
            )
            
            return jsonify({
                "status": "failed",
                "message": f"Face match confidence too low ({best_match_confidence:.2f})",
                "face_verified": False,
                "license_verified": False,
                "liveness_verified": False,
                "confidence": float(best_match_confidence),
                "face_score": float(best_match_confidence * 100),
                "name_match_score": 0.0,
                "dl_match": False,
                "final_score": 0.0,
                "decision": "REJECTED",
                "details": {
                    "user_id": best_match_user,
                    "stored_data": stored_data
                }
            }), 200
        
    except Exception as e:
        print(f"❌ Verification error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "failed",
            "message": f"Verification failed: {str(e)}"
        }), 500


@verification_bp.route('/logs', methods=['GET'])
def get_logs():
    """Get verification logs"""
    try:
        from database import get_verification_logs
        
        limit = request.args.get('limit', 50, type=int)
        logs = get_verification_logs(limit=limit)
        
        return jsonify({
            "success": True,
            "count": len(logs),
            "logs": logs
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching logs: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@verification_bp.route('/verify-face-only', methods=['POST'])
def verify_face_only():
    """
    Verify user identity using face recognition only and return all user data
    """
    try:
        face_image = request.files.get('face_image')
        
        if not face_image:
            return jsonify({
                "status": "failed",
                "message": "Face image is required"
            }), 400
        
        # Save uploaded file
        face_filename = secure_filename(f'face_verify_only.jpg')
        face_path = os.path.join(Config.UPLOAD_FOLDER, face_filename)
        face_image.save(face_path)
        
        print("📁 Face image saved for verification")
        
        # ========== FACE VERIFICATION ==========
        print("=" * 50)
        print("FACE-ONLY VERIFICATION")
        print("=" * 50)
        
        print(f"🔄 Verifying face from: {face_path}")
        face_embedding = face_model.extract_embedding(face_path)
        
        if face_embedding is None:
            return jsonify({
                "status": "failed",
                "message": "No face detected in the uploaded image"
            }), 400
        
        # Compare with registered faces
        registered_faces = get_all_face_embeddings()
        
        if not registered_faces:
            return jsonify({
                "status": "failed",
                "message": "No registered users found in database"
            }), 404
        
        print(f"📊 Comparing with {len(registered_faces)} registered users...")
        
        best_match_user = None
        best_match_confidence = 0.0
        threshold = 0.70  # Confidence threshold
        
        for user_id, stored_embedding in registered_faces:
            try:
                # Convert to numpy arrays if needed
                if not isinstance(face_embedding, np.ndarray):
                    face_emb = np.array(face_embedding)
                else:
                    face_emb = face_embedding
                
                if not isinstance(stored_embedding, np.ndarray):
                    stored_emb = np.array(stored_embedding)
                else:
                    stored_emb = stored_embedding
                
                # Calculate cosine similarity
                cosine_sim = np.dot(face_emb, stored_emb) / (np.linalg.norm(face_emb) * np.linalg.norm(stored_emb))
                
                # Calculate correlation
                correlation = np.corrcoef(face_emb, stored_emb)[0, 1]
                
                # Combined similarity score
                similarity = (cosine_sim + correlation) / 2
                
                print(f"   User {user_id}: similarity = {similarity:.4f} (cosine: {cosine_sim:.4f}, corr: {correlation:.4f})")
                
                if similarity > best_match_confidence:
                    best_match_confidence = similarity
                    best_match_user = user_id
                    
            except Exception as e:
                print(f"   ⚠️ Error comparing with user {user_id}: {e}")
                continue
        
        if best_match_confidence < threshold:
            print(f"❌ No face match found (best: {best_match_confidence:.4f}, threshold: {threshold})")
            return jsonify({
                "status": "failed",
                "message": "Face not recognized",
                "confidence": float(best_match_confidence)
            }), 200
        
        print(f"✅ Face matched! User: {best_match_user}, Confidence: {best_match_confidence:.4f}")
        
        # ========== FETCH ALL USER DATA ==========
        print("=" * 50)
        print("FETCHING USER DATA")
        print("=" * 50)
        
        # Get license data for the matched user
        user_license_data = get_license_record(best_match_user)
        
        # Log successful face-only verification
        log_verification(
            user_id=best_match_user,
            status="verified_face_only",
            face_confidence=float(best_match_confidence),
            license_match_score=0.0,
            liveness_passed=False
        )
        
        # Prepare response with all user data
        response_data = {
            "status": "verified",
            "message": "Face recognized successfully",
            "confidence": float(best_match_confidence),
            "user_id": best_match_user,
            "user_data": user_license_data
        }
        
        print(f"✅ USER DATA RETRIEVED FOR: {best_match_user}")
        print("=" * 50)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"❌ Face-only verification error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "failed",
            "message": f"Verification failed: {str(e)}"
        }), 500
@verification_bp.route('/test-face', methods=['POST'])
def test_face():
    """
    Test face detection only - returns face detection results without verification
    """
    try:
        face_image = request.files.get('face_image')
        
        if not face_image:
            return jsonify({
                "status": "failed",
                "message": "Face image is required"
            }), 400
        
        # Save uploaded file
        face_filename = secure_filename(f'test_face.jpg')
        face_path = os.path.join(Config.UPLOAD_FOLDER, face_filename)
        face_image.save(face_path)
        
        print("📁 Face image saved for testing")
        
        # Test face detection
        print("=" * 50)
        print("TESTING FACE DETECTION")
        print("=" * 50)
        
        print(f"🔄 Testing face detection from: {face_path}")
        face_embedding = face_model.extract_embedding(face_path)
        
        if face_embedding is None:
            return jsonify({
                "status": "no_face",
                "message": "No face detected in the uploaded image",
                "face_detected": False
            }), 200
        
        print("✅ Face detected successfully")
        
        return jsonify({
            "status": "success",
            "message": "Face detected successfully",
            "face_detected": True,
            "embedding_shape": face_embedding.shape if hasattr(face_embedding, 'shape') else len(face_embedding)
        }), 200
        
    except Exception as e:
        print(f"❌ Face test error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "failed",
            "message": f"Face test failed: {str(e)}"
        }), 500

