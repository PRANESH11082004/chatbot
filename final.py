import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from datetime import datetime
import hashlib
from google import genai
import os
from dotenv import load_dotenv
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
from gtts import gTTS
import base64



# Load environment variables
load_dotenv()

# ---------------------------
# MongoDB Configuration
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "legal_chatbot")

# Admin credentials
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "admin@123"

@st.cache_resource
def get_mongo_client():
    """Initialize MongoDB client"""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# ---------------------------
# Gemini API Configuration - Multiple Keys
# ---------------------------
def load_gemini_api_keys():
    """Load all Gemini API keys from environment"""
    keys = []
    for i in range(1, 11):  # Load 10 keys
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    return keys

GEMINI_API_KEYS = load_gemini_api_keys()

def get_active_api_key_index():
    """Get the current active API key index from MongoDB"""
    client = get_mongo_client()
    if not client:
        return 0

    db = client[DB_NAME]
    config_collection = db.api_config

    config = config_collection.find_one({"type": "gemini_api"})
    if config:
        return config.get("active_key_index", 0)
    else:
        # Initialize with first key
        config_collection.insert_one({"type": "gemini_api", "active_key_index": 0})
        return 0

def set_active_api_key_index(index):
    """Set the active API key index in MongoDB"""
    client = get_mongo_client()
    if not client:
        return False

    db = client[DB_NAME]
    config_collection = db.api_config

    config_collection.update_one(
        {"type": "gemini_api"},
        {"$set": {"active_key_index": index}},
        upsert=True
    )
    # Clear the cached client
    get_gemini_client.clear()
    return True

def rotate_api_key():
    """Rotate to the next API key"""
    current_index = get_active_api_key_index()
    next_index = (current_index + 1) % len(GEMINI_API_KEYS)
    set_active_api_key_index(next_index)
    print(f"Rotated API key from index {current_index} to {next_index}")
    return next_index

@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client with the current active key"""
    if not GEMINI_API_KEYS:
        st.error("No Gemini API keys found in environment variables!")
        return None

    active_index = get_active_api_key_index()
    active_key = GEMINI_API_KEYS[active_index]
    print(f"Using Gemini API Key #{active_index + 1}")
    return genai.Client(api_key=active_key)

# ---------------------------
# Authentication Functions
# ---------------------------
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register new user in MongoDB"""
    client = get_mongo_client()
    if not client:
        return False, "Database connection failed" 
    
    db = client[DB_NAME]
    users_collection = db.users
    
    # Check if user already exists
    if users_collection.find_one({"username": username}):
        return False, "Username already exists"
    
    if users_collection.find_one({"email": email}):
        return False, "Email already registered"
    
    # Create new user
    user_data = {
        "username": username,
        "email": email,
        "password": hash_password(password),
        "created_at": datetime.now(),
        "last_login": None
    }
    
    users_collection.insert_one(user_data)
    return True, "Registration successful!"

def apply_global_styles():
    """Unified Professional UI with Zero-Indentation st.html"""
    st.html("""
<style>
/* Global Background and Modern Font */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism: Cards and Chat Bubbles */
[data-testid="stForm"], [data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    transition: all 0.4s ease;
}

/* Hover Effects */
[data-testid="stForm"]:hover, [data-testid="stChatMessage"]:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(0, 242, 254, 0.4) !important;
    box-shadow: 0 15px 35px rgba(0, 242, 254, 0.2);
}

/* Modern Button Styling */
.stButton>button {
    width: 100%;
    border-radius: 30px;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white !important;
    border: none;
    font-weight: 700;
    transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.stButton>button:hover {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: #000000 !important;
    transform: scale(1.05);
}
</style>
    """)

def text_to_speech(text):
    """Converts text to speech and returns an HTML audio player"""
    try:
        # Create a TTS object
        tts = gTTS(text=text, lang='en')
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            
            # Read the file and encode to base64 for Streamlit
            with open(fp.name, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                
                # HTML for an auto-playing audio element
                audio_html = f"""
                    <audio controls autoplay="true">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                    """
                return audio_html
    except Exception as e:
        return f"Error generating audio: {str(e)}"

def login_user(username, password):
    """Authenticate user"""
    # Check if admin
    if is_admin(username, password):
        return True, {
            "username": username,
            "email": username,
            "is_admin": True
        }

    client = get_mongo_client()
    if not client:
        return False, None

    db = client[DB_NAME]
    users_collection = db.users

    user = users_collection.find_one({
        "username": username,
        "password": hash_password(password)
    })

    if user:
        # Update last login
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.now()}}
        )
        user["is_admin"] = False
        return True, user

    return False, None

def save_chat_history(username, category, user_message, bot_response):
    """Save chat interaction to MongoDB"""
    client = get_mongo_client()
    if not client:
        return False
    
    db = client[DB_NAME]
    chat_collection = db.chat_history
    
    chat_data = {
        "username": username,
        "category": category,
        "user_message": user_message,
        "bot_response": bot_response,
        "timestamp": datetime.now()
    }
    
    chat_collection.insert_one(chat_data)
    return True

def get_user_chat_history(username, limit=50):
    """Retrieve user's chat history"""
    client = get_mongo_client()
    if not client:
        return []

    db = client[DB_NAME]
    chat_collection = db.chat_history

    chats = chat_collection.find(
        {"username": username}
    ).sort("timestamp", -1).limit(limit)

    return list(chats)

# ---------------------------
# Voice-to-Text Functions
# ---------------------------
def audio_to_text(audio_file):
    """Convert audio file to text using speech recognition"""
    try:
        recognizer = sr.Recognizer()

        # Get file extension
        file_extension = audio_file.name.split('.')[-1].lower()

        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_input:
            tmp_input.write(audio_file.read())
            tmp_input_path = tmp_input.name

        # Convert to WAV if not already WAV
        wav_path = tmp_input_path
        if file_extension != 'wav':
            try:
                audio = AudioSegment.from_file(tmp_input_path, format=file_extension)
                wav_path = tmp_input_path.replace(f'.{file_extension}', '.wav')
                audio.export(wav_path, format='wav')
            except Exception as conv_error:
                os.unlink(tmp_input_path)
                return False, f"Error converting audio format: {str(conv_error)}"

        # Convert audio to text
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # Clean up temp files
        os.unlink(tmp_input_path)
        if wav_path != tmp_input_path:
            os.unlink(wav_path)

        return True, text
    except sr.UnknownValueError:
        return False, "Could not understand the audio. Please try again with clearer speech."
    except sr.RequestError as e:
        return False, f"Could not request results from speech recognition service: {e}"
    except Exception as e:
        return False, f"Error processing audio: {str(e)}"

# ---------------------------
# Quiz Management Functions
# ---------------------------
def is_admin(username, password):
    """Check if user is admin"""
    return username == ADMIN_EMAIL and password == ADMIN_PASSWORD

def create_quiz(quiz_data):
    """Create a new quiz"""
    client = get_mongo_client()
    if not client:
        return False

    db = client[DB_NAME]
    quiz_collection = db.quizzes

    quiz_data["created_at"] = datetime.now()
    quiz_collection.insert_one(quiz_data)
    return True

def get_all_quizzes():
    """Get all quizzes"""
    client = get_mongo_client()
    if not client:
        return []

    db = client[DB_NAME]
    quiz_collection = db.quizzes

    return list(quiz_collection.find().sort("created_at", -1))

def get_quiz_by_id(quiz_id):
    """Get quiz by ID"""
    from bson.objectid import ObjectId
    client = get_mongo_client()
    if not client:
        return None

    db = client[DB_NAME]
    quiz_collection = db.quizzes

    return quiz_collection.find_one({"_id": ObjectId(quiz_id)})

def delete_quiz(quiz_id):
    """Delete a quiz"""
    from bson.objectid import ObjectId
    client = get_mongo_client()
    if not client:
        return False

    db = client[DB_NAME]
    quiz_collection = db.quizzes

    result = quiz_collection.delete_one({"_id": ObjectId(quiz_id)})
    return result.deleted_count > 0

def save_quiz_result(username, quiz_id, score, total_questions, answers):
    """Save quiz result"""
    client = get_mongo_client()
    if not client:
        return False

    db = client[DB_NAME]
    results_collection = db.quiz_results

    # Convert integer keys to strings for MongoDB compatibility
    answers_str_keys = {str(k): v for k, v in answers.items()}

    result_data = {
        "username": username,
        "quiz_id": quiz_id,
        "score": score,
        "total_questions": total_questions,
        "percentage": (score / total_questions) * 100,
        "answers": answers_str_keys,
        "timestamp": datetime.now()
    }

    results_collection.insert_one(result_data)
    return True

def get_user_quiz_results(username):
    """Get user's quiz results"""
    client = get_mongo_client()
    if not client:
        return []

    db = client[DB_NAME]
    results_collection = db.quiz_results

    return list(results_collection.find({"username": username}).sort("timestamp", -1))

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Legal Advisor",

)

# ---------------------------
# Authentication UI
# ---------------------------
def show_auth_page():
    """Display login/register page"""
    apply_global_styles()
    st.markdown(
        "<h1 style='text-align: center;'>Legal Advisor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>Your AI-powered legal consultation assistant</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", use_container_width=True):
            if login_username and login_password:
                success, user = login_user(login_username, login_password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.session_state.user_data = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register", use_container_width=True):
            if reg_username and reg_email and reg_password and reg_confirm:
                if reg_password != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    success, message = register_user(reg_username, reg_password, reg_email)
                    if success:
                        st.success(message)
                        st.info("Please login with your credentials")
                    else:
                        st.error(message)
            else:
                st.warning("Please fill all fields")



# ---------------------------
# Load Legal Datasets from JSON
# ---------------------------
@st.cache_data
def load_legal_datasets():
    """Load legal datasets from JSON file"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'legal_datasets.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Legal datasets file not found. Please ensure 'legal_datasets.json' exists in the project directory.")
        return {}
    except json.JSONDecodeError:
        st.error("Error parsing legal datasets JSON file.")
        return {}

LEGAL_DATASETS = load_legal_datasets()

# ---------------------------
# Load ML Model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
gemini_client = get_gemini_client()

# ---------------------------
# Query Gemini API
# ---------------------------
def query_gemini(user_question, category, retry_count=0):
    """Query Gemini API when no match found in dataset"""
    try:
        # Enhanced single prompt that validates and responds
        prompt = f"""You are a legal assistant specializing in Indian law. This question has been categorized under {category}.

CRITICAL INSTRUCTIONS:
1. First, determine if this question is about law or legal matters (Constitutional Law, Civil Law, Environmental Law, or Labor Law)
2. If the question is about technology, products, entertainment, food, shopping, or any non-legal topic, respond EXACTLY with: "⚖️ I can only answer questions about Indian law. Please ask about Constitutional Law, Civil Law, Environmental Law, or Labor Law."
3. If the question IS about law, provide a helpful legal answer focused on {category} (max 150 words)

Question: {user_question}

Response:"""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()
        else:
            return f"⚠️ No response generated. Please try again."

    except Exception as e:
        error_msg = str(e).lower()

        # Check if error is due to quota/rate limit
        if ("quota" in error_msg or "rate" in error_msg or "limit" in error_msg or "429" in error_msg) and retry_count < len(GEMINI_API_KEYS):
            print(f"API Key quota exceeded. Rotating to next key... (Attempt {retry_count + 1})")
            rotate_api_key()
            # Retry with next key
            return query_gemini(user_question, category, retry_count + 1)
        else:
            return f"⚠️ Unable to generate response. Error: {str(e)}"

# ---------------------------
# Intent Detection
# ---------------------------
def detect_harmful_intent(user_question, retry_count=0):
    """Detect if the user's question has harmful or malicious intent"""
    try:
        prompt = f"""You are a safety classifier for a legal advisory chatbot. Analyze the user's question and determine if it has harmful or malicious intent.

HARMFUL INTENT includes:
- Asking how to commit crimes or illegal activities
- Seeking ways to harm others (physically, financially, emotionally)
- Trying to exploit legal loopholes for illegal purposes
- Planning fraud, scams, or deception
- Evading justice or hiding criminal activity
- Harassing, threatening, or stalking someone
- Asking about illegal substances, weapons for harm, etc.

LEGITIMATE INTENT includes:
- Asking about legal rights and protections
- Understanding laws and legal procedures
- Seeking information about legal consequences
- Academic or educational questions about law
- Understanding how to legally protect oneself
- Questions about victim rights or how to report crimes

User Question: {user_question}

Respond with ONLY ONE WORD:
- "SAFE" if the question has legitimate legal intent
- "HARMFUL" if the question has malicious or harmful intent

Response:"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )

        if response and response.text:
            result = response.text.strip().upper()
            is_safe = "SAFE" in result
            print(f"Intent Detection: {result} - Question: {user_question[:50]}...")
            return is_safe
        else:
            # If unsure, allow (fail open for better UX, but log it)
            print("Intent detection failed - defaulting to SAFE")
            return True

    except Exception as e:
        error_msg = str(e).lower()

        # Check if error is due to quota/rate limit
        if ("quota" in error_msg or "rate" in error_msg or "limit" in error_msg or "429" in error_msg) and retry_count < len(GEMINI_API_KEYS):
            print(f"API Key quota exceeded. Rotating to next key... (Attempt {retry_count + 1})")
            rotate_api_key()
            # Retry with next key
            return detect_harmful_intent(user_question, retry_count + 1)
        else:
            print(f"Intent detection error: {str(e)}")
            # Fail open - allow question if detection fails
            return True

# ---------------------------
# Auto-Classify Category
# ---------------------------
def classify_category(user_input):
    """Automatically classify the legal category based on user question"""
    if not LEGAL_DATASETS:
        return None, 0.0

    category_scores = {}

    # For each category, find the best matching question
    for category, dataset in LEGAL_DATASETS.items():
        if not dataset:
            continue

        prompts = [item["prompt"] for item in dataset]
        user_emb = model.encode(user_input, convert_to_tensor=True)
        prompt_embs = model.encode(prompts, convert_to_tensor=True)
        similarities = util.cos_sim(user_emb, prompt_embs)[0]

        # Store the highest similarity for this category
        category_scores[category] = float(similarities.max())

    if not category_scores:
        return None, 0.0

    # Return category with highest score
    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]

    print(f"Category classification: {best_category} (confidence: {best_score:.3f})")
    print(f"All scores: {category_scores}")

    return best_category, best_score

# ---------------------------
# Find Answer Function
# ---------------------------
def find_answer(user_input, category, threshold=0.85):
    """Find answer from dataset or use Gemini API"""
    dataset = LEGAL_DATASETS.get(category, [])

    if not dataset:
        return "⚠️ Invalid category selected.", category

    prompts = [item["prompt"] for item in dataset]
    completions = [item["completion"] for item in dataset]

    user_emb = model.encode(user_input, convert_to_tensor=True)
    prompt_embs = model.encode(prompts, convert_to_tensor=True)
    similarities = util.cos_sim(user_emb, prompt_embs)[0]

    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    # Check for different years or updated acts
    user_lower = user_input.lower()
    import re
    user_years = re.findall(r'\b(19|20)\d{2}\b', user_lower)
    dataset_years = re.findall(r'\b(19|20)\d{2}\b', prompts[best_idx].lower())

    # If user asks about different year than in dataset, force Gemini API
    if user_years and dataset_years and user_years[0] != dataset_years[0]:
        print(f"Year mismatch: User asked {user_years[0]}, dataset has {dataset_years[0]} - Using Gemini API")
        return "🤖 " + query_gemini(user_input, category), category

    # Check for words like "new", "updated", "latest", "current" that suggest different info needed
    trigger_words = ['new', 'updated', 'latest', 'current', 'recent', 'modern']
    if any(word in user_lower for word in trigger_words):
        print(f"Trigger word found - Using Gemini API")
        return "🤖 " + query_gemini(user_input, category), category

    # Debug info (remove in production)
    print(f"Question: {user_input}")
    print(f"Best match: {prompts[best_idx]}")
    print(f"Similarity score: {best_score:.3f}")
    print(f"Threshold: {threshold}")

    if best_score >= threshold:
        print("Using dataset answer")
        return completions[best_idx], category
    else:
        # No match found - query Gemini API
        print("Using Gemini API")
        return "🤖 " + query_gemini(user_input, category), category

# ---------------------------
# Admin Quiz Management UI
# ---------------------------
def admin_quiz_panel():
    """Admin panel for creating and managing quizzes"""
    st.markdown("## 👨‍💼 Admin Panel")

    tab1, tab2, tab3 = st.tabs(["Create Quiz", "Manage Quizzes", "API Key Management"])

    with tab1:
        st.subheader("Create New Quiz")

        quiz_title = st.text_input("Quiz Title", key="quiz_title")
        quiz_category = st.selectbox("Category", list(LEGAL_DATASETS.keys()), key="quiz_category")
        quiz_description = st.text_area("Description", key="quiz_description")

        st.markdown("---")
        st.markdown("### Add Questions")

        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = []

        # Form to add questions
        with st.form("add_question_form"):
            question_text = st.text_area("Question")
            option_a = st.text_input("Option A")
            option_b = st.text_input("Option B")
            option_c = st.text_input("Option C")
            option_d = st.text_input("Option D")
            correct_answer = st.selectbox("Correct Answer", ["A", "B", "C", "D"])
            explanation = st.text_area("Explanation for Correct Answer", help="Explain why this answer is correct. This will be shown to users after they complete the quiz.")

            if st.form_submit_button("Add Question"):
                if question_text and option_a and option_b and option_c and option_d and explanation:
                    question = {
                        "question": question_text,
                        "options": {
                            "A": option_a,
                            "B": option_b,
                            "C": option_c,
                            "D": option_d
                        },
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    }
                    st.session_state.quiz_questions.append(question)
                    st.success(f"Question {len(st.session_state.quiz_questions)} added!")
                    st.rerun()
                else:
                    st.error("Please fill all fields including the explanation")

        # Display added questions
        if st.session_state.quiz_questions:
            st.markdown(f"**Questions Added: {len(st.session_state.quiz_questions)}**")
            for idx, q in enumerate(st.session_state.quiz_questions):
                with st.expander(f"Question {idx + 1}: {q['question'][:50]}..."):
                    st.write(f"**Q:** {q['question']}")
                    for key, value in q['options'].items():
                        marker = "✅" if key == q['correct_answer'] else ""
                        st.write(f"{key}. {value} {marker}")
                    st.info(f"**Explanation:** {q.get('explanation', 'No explanation provided')}")
                    if st.button(f"Remove Question {idx + 1}", key=f"remove_{idx}"):
                        st.session_state.quiz_questions.pop(idx)
                        st.rerun()

        # Create quiz button
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Quiz", use_container_width=True, type="primary"):
                if quiz_title and quiz_description and st.session_state.quiz_questions:
                    quiz_data = {
                        "title": quiz_title,
                        "category": quiz_category,
                        "description": quiz_description,
                        "questions": st.session_state.quiz_questions
                    }
                    if create_quiz(quiz_data):
                        st.success("Quiz created successfully!")
                        st.session_state.quiz_questions = []
                        st.rerun()
                    else:
                        st.error("Failed to create quiz")
                else:
                    st.error("Please provide title, description and at least one question")
        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.quiz_questions = []
                st.rerun()

    with tab2:
        st.subheader("Manage Existing Quizzes")

        quizzes = get_all_quizzes()

        if not quizzes:
            st.info("No quizzes created yet")
        else:
            for quiz in quizzes:
                with st.expander(f"📝 {quiz['title']} ({quiz['category']})"):
                    st.write(f"**Description:** {quiz['description']}")
                    st.write(f"**Questions:** {len(quiz['questions'])}")
                    st.write(f"**Created:** {quiz['created_at'].strftime('%Y-%m-%d %H:%M')}")

                    for idx, q in enumerate(quiz['questions']):
                        st.markdown(f"**Q{idx + 1}:** {q['question']}")
                        for key, value in q['options'].items():
                            marker = "✅" if key == q['correct_answer'] else ""
                            st.write(f"  {key}. {value} {marker}")
                        st.info(f"**Explanation:** {q.get('explanation', 'No explanation provided')}")
                        st.markdown("---")

                    if st.button(f"Delete Quiz", key=f"delete_{quiz['_id']}"):
                        if delete_quiz(str(quiz['_id'])):
                            st.success("Quiz deleted!")
                            st.rerun()

    with tab3:
        st.subheader("🔑 Gemini API Key Management")

        # Display current API key info
        current_index = get_active_api_key_index()
        total_keys = len(GEMINI_API_KEYS)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total API Keys", total_keys)
        with col2:
            st.metric("Active Key", f"Key #{current_index + 1}")
        with col3:
            st.metric("Remaining Keys", total_keys - current_index - 1)

        st.markdown("---")

        # Current key display (masked)
        if GEMINI_API_KEYS:
            active_key = GEMINI_API_KEYS[current_index]
            masked_key = active_key[:10] + "..." + active_key[-4:]
            st.info(f"🔐 **Current API Key (Key #{current_index + 1}):** `{masked_key}`")
        else:
            st.error("No API keys loaded!")

        st.markdown("---")

        # Manual key rotation
        st.markdown("### Manual Key Management")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Rotate to Next Key**")
            st.write("Switch to the next available API key in rotation.")
            if st.button("🔄 Rotate to Next Key", use_container_width=True, type="primary"):
                new_index = rotate_api_key()
                st.success(f"✅ Rotated to API Key #{new_index + 1}")
                st.rerun()

        with col2:
            st.markdown("**Select Specific Key**")
            selected_key = st.selectbox(
                "Choose API Key:",
                options=list(range(1, total_keys + 1)),
                index=current_index,
                format_func=lambda x: f"API Key #{x}"
            )
            if st.button("✓ Switch to Selected Key", use_container_width=True):
                if set_active_api_key_index(selected_key - 1):
                    st.success(f"✅ Switched to API Key #{selected_key}")
                    st.rerun()
                else:
                    st.error("Failed to switch API key")

        st.markdown("---")

        # Key list display
        st.markdown("### All API Keys")
        for idx, key in enumerate(GEMINI_API_KEYS):
            masked = key[:10] + "..." + key[-4:]
            is_active = idx == current_index
            status = "🟢 ACTIVE" if is_active else "⚪ Available"
            st.text(f"{status} | Key #{idx + 1}: {masked}")

        st.markdown("---")
        st.info("💡 **Tip:** API keys automatically rotate when quota is exhausted. You can also manually rotate keys here if needed.")

# ---------------------------
# User Quiz UI
# ---------------------------
def user_quiz_panel():
    """User interface for taking quizzes"""
    st.markdown("## 📝 Legal Knowledge Quiz")

    tab1, tab2 = st.tabs(["Available Quizzes", "My Results"])

    with tab1:
        # Check if quiz results should be shown
        if "show_quiz_results" in st.session_state and st.session_state.show_quiz_results:
            # Show results
            result = st.session_state.quiz_result
            st.markdown(f"### 🎯 Quiz Results: {result['quiz_title']}")
            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{result['score']}/{result['total']}")
            with col2:
                st.metric("Percentage", f"{result['percentage']:.1f}%")
            with col3:
                if result['percentage'] >= 80:
                    st.metric("Grade", "Excellent 🎉")
                elif result['percentage'] >= 60:
                    st.metric("Grade", "Good 👍")
                else:
                    st.metric("Grade", "Keep Practicing 📚")

            st.markdown("---")

            if result['percentage'] >= 80:
                st.balloons()
                st.success("🎉 Excellent performance!")
            elif result['percentage'] >= 60:
                st.info("👍 Good job!")
            else:
                st.warning("📚 Keep practicing!")

            # Show detailed results with explanations
            st.markdown("---")
            st.markdown("### 📋 Detailed Results")

            quiz = st.session_state.active_quiz
            user_answers = st.session_state.quiz_answers

            for idx, question in enumerate(quiz['questions']):
                user_answer = user_answers.get(idx, "No answer")
                correct_answer = question['correct_answer']
                is_correct = user_answer == correct_answer

                with st.expander(f"Question {idx + 1}: {'✅ Correct' if is_correct else '❌ Incorrect'}"):
                    st.markdown(f"**Q:** {question['question']}")
                    st.markdown("**Options:**")
                    for key, value in question['options'].items():
                        if key == user_answer and key == correct_answer:
                            st.success(f"{key}. {value} ✅ (Your answer - Correct!)")
                        elif key == user_answer:
                            st.error(f"{key}. {value} ❌ (Your answer)")
                        elif key == correct_answer:
                            st.success(f"{key}. {value} ✅ (Correct answer)")
                        else:
                            st.write(f"{key}. {value}")

                    st.info(f"**💡 Explanation:** {question.get('explanation', 'No explanation available')}")

            # Back button (outside form)
            st.markdown("---")
            if st.button("Back to Quiz List", use_container_width=True, type="primary"):
                del st.session_state.show_quiz_results
                del st.session_state.quiz_result
                if "active_quiz" in st.session_state:
                    del st.session_state.active_quiz
                if "quiz_answers" in st.session_state:
                    del st.session_state.quiz_answers
                st.rerun()

        # Check if a quiz is active
        elif "active_quiz" in st.session_state and st.session_state.active_quiz:
            # Show only the active quiz
            quiz = st.session_state.active_quiz
            st.markdown(f"### 🎯 {quiz['title']}")
            st.write(f"**Category:** {quiz['category']}")
            st.write(f"**Description:** {quiz['description']}")
            st.markdown("---")

            with st.form("quiz_form"):
                for idx, question in enumerate(quiz['questions']):
                    st.markdown(f"**Question {idx + 1}:** {question['question']}")
                    answer = st.radio(
                        "Select your answer:",
                        options=list(question['options'].keys()),
                        format_func=lambda x, q=question: f"{x}. {q['options'][x]}",
                        key=f"q_{idx}"
                    )
                    st.session_state.quiz_answers[idx] = answer
                    st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("Submit Quiz", type="primary", use_container_width=True)
                with col2:
                    cancel = st.form_submit_button("Cancel", use_container_width=True)

                if cancel:
                    del st.session_state.active_quiz
                    del st.session_state.quiz_answers
                    st.rerun()

                if submitted:
                    # Calculate score
                    score = 0
                    for idx, question in enumerate(quiz['questions']):
                        if st.session_state.quiz_answers.get(idx) == question['correct_answer']:
                            score += 1

                    # Save result
                    save_quiz_result(
                        st.session_state.username,
                        str(quiz['_id']),
                        score,
                        len(quiz['questions']),
                        st.session_state.quiz_answers
                    )

                    # Store results and trigger results view
                    percentage = (score/len(quiz['questions'])*100)
                    st.session_state.quiz_result = {
                        "quiz_title": quiz['title'],
                        "score": score,
                        "total": len(quiz['questions']),
                        "percentage": percentage
                    }
                    st.session_state.show_quiz_results = True
                    st.rerun()
        else:
            # Show quiz list
            quizzes = get_all_quizzes()

            if not quizzes:
                st.info("No quizzes available yet. Check back later!")
            else:
                st.markdown("### 📚 Available Quizzes")
                st.write("Click on a quiz to start!")
                st.markdown("---")

                for quiz in quizzes:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**📝 {quiz['title']}**")
                        st.write(f"*Category:* {quiz['category']}")
                        st.write(f"*Description:* {quiz['description']}")
                        st.write(f"*Questions:* {len(quiz['questions'])}")
                    with col2:
                        if st.button("Start Quiz", key=f"start_{quiz['_id']}", use_container_width=True):
                            st.session_state.active_quiz = quiz
                            st.session_state.quiz_answers = {}
                            st.rerun()
                    st.markdown("---")

    with tab2:
        st.subheader("Your Quiz Results")
        results = get_user_quiz_results(st.session_state.username)

        if not results:
            st.info("You haven't taken any quizzes yet")
        else:
            for result in results:
                quiz = get_quiz_by_id(result['quiz_id'])
                quiz_title = quiz['title'] if quiz else "Unknown Quiz"

                with st.expander(f"📊 {quiz_title} - {result['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    st.metric("Score", f"{result['score']}/{result['total_questions']}")
                    st.metric("Percentage", f"{result['percentage']:.1f}%")

                    if result['percentage'] >= 80:
                        st.success("Excellent! 🎉")
                    elif result['percentage'] >= 60:
                        st.info("Good job! 👍")
                    else:
                        st.warning("Keep practicing! 📚")

# ---------------------------
# Main App
# ---------------------------
def main_app():
    """Main application after login"""
    apply_global_styles()

    # Your existing code continues...
    is_admin_user = st.session_state.user_data.get("is_admin", False)

    # Check if user is admin
    is_admin_user = st.session_state.user_data.get("is_admin", False)

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align: center;'>Legal Advisor</h1>",
            unsafe_allow_html=True
        )

    # Logout button in sidebar
    with st.sidebar:
        if is_admin_user:
            st.markdown(f"### 👨‍💼 Admin: {st.session_state.username}")
        else:
            st.markdown(f"### 👤 {st.session_state.username}")

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        # Info about features
        st.markdown("### 🤖 Smart Features")
        st.info("✨ **Auto Category Detection**: The system automatically detects the relevant legal category.\n\n🛡️ **Safety Protection**: AI-powered intent detection prevents harmful or illegal queries.")

        # Available categories
        st.markdown("### 📚 Available Categories")
        st.success("⚖️ Constitutional Law\n\n📜 Civil Law\n\n🌿 Environmental Law\n\n👷 Labor Law")

        st.divider()

        # Clear chat button (for all users)
        st.markdown("### 🗑️ Chat Actions")
        if st.button("Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            st.rerun()

    # Main content area with tabs
    if is_admin_user:
        # Admin view - Chat, Quiz, and Admin Panel
        tab1, tab2, tab3 = st.tabs(["💬 Legal Chat", "📝 Quiz", "👨‍💼 Admin Panel"])

        with tab1:
            # Chat interface
            chat_interface()

        with tab2:
            # Quiz interface (take quizzes)
            user_quiz_panel()

        with tab3:
            # Admin panel (manage quizzes and API keys)
            admin_quiz_panel()
    else:
        # Regular user view - Chat and Quiz
        tab1, tab2 = st.tabs(["💬 Legal Chat", "📝 Quiz"])

        with tab1:
            # Chat interface (existing code)
            chat_interface()

        with tab2:
            # Quiz interface
            user_quiz_panel()

def chat_interface():
    """Chat interface for regular users"""

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize voice input counter for resetting audio recorder
    if "voice_input_counter" not in st.session_state:
        st.session_state.voice_input_counter = 0

    # Display chat history
    for i, (role, msg) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.markdown(msg)
            
            # Add the Listen button for assistant responses
            if role == "assistant":
                if st.button(f"🔊 Listen", key=f"tts_btn_{i}"):
                    # 1. Get the user's question (the message right before this one)
                    user_question = st.session_state.chat_history[i-1][1] if i > 0 else ""
                    
                    # 2. Extract only the explanation if a header exists
                    # This removes the "⚖️ Constitutional Law" part from the speech
                    if "\n\n" in msg:
                        explanation_only = msg.split("\n\n", 1)[1]
                        full_speech_text = f"The question was: {user_question}. The answer is: {explanation_only}"
                    else:
                        full_speech_text = f"The question was: {user_question}. The answer is: {msg}"
                    
                    # 3. Clean formatting (remove bold and headers)
                    full_speech_text = full_speech_text.replace("**", "").replace("#", "")
                    
                    # 4. Remove Emojis so they aren't spoken (e.g., "Scroll")
                    emojis_to_remove = ["⚖️", "📜", "🌿", "👷", "📚", "🤖"]
                    for emoji in emojis_to_remove:
                        full_speech_text = full_speech_text.replace(emoji, "")
                    
                    # 5. Generate and play audio
                    audio_html = text_to_speech(full_speech_text)
                    st.markdown(audio_html, unsafe_allow_html=True)

    # Voice-to-Text Section
    st.markdown("---")
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown("##### 🎤 Voice Input (Optional)")
    with col_clear:
        if st.button("🔄 Clear", key="clear_voice", help="Clear voice input and start fresh"):
            if "transcribed_text" in st.session_state:
                del st.session_state.transcribed_text
            if "edit_mode" in st.session_state:
                del st.session_state.edit_mode
            # Increment counter to reset audio recorder
            st.session_state.voice_input_counter += 1
            st.rerun()

    # Create tabs for recording and uploading
    voice_tab1, voice_tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio File"])

    user_input = None

    with voice_tab1:
        st.markdown("**Record your question directly in the browser:**")
        audio_recorded = st.audio_input("Click to start recording", key=f"audio_recorder_{st.session_state.voice_input_counter}")

        if audio_recorded is not None:
            # Auto-convert to text if not already converted
            if "transcribed_text" not in st.session_state:
                with st.spinner("Converting audio to text..."):
                    success, result = audio_to_text(audio_recorded)
                    if success:
                        st.session_state.transcribed_text = result
                        st.rerun()
                    else:
                        st.error(result)
                        # Show discard button on error
                        if st.button("🗑️ Try Again", use_container_width=True, key="discard_error"):
                            st.session_state.voice_input_counter += 1
                            st.rerun()

    with voice_tab2:
        st.markdown("**Upload a pre-recorded audio file:**")
        audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'ogg', 'flac'], key=f"audio_upload_{st.session_state.voice_input_counter}", label_visibility="collapsed")

        if audio_file is not None:
            st.audio(audio_file, format=f'audio/{audio_file.type.split("/")[1]}')
            # Auto-convert to text if not already converted
            if "transcribed_text" not in st.session_state:
                with st.spinner("Converting audio to text..."):
                    success, result = audio_to_text(audio_file)
                    if success:
                        st.session_state.transcribed_text = result
                        st.rerun()
                    else:
                        st.error(result)
                        # Show try again button on error
                        if st.button("🗑️ Try Again", use_container_width=True, key="discard_upload_error"):
                            st.session_state.voice_input_counter += 1
                            st.rerun()

    # Display transcribed text if available
    if "transcribed_text" in st.session_state and st.session_state.transcribed_text:
        st.markdown("---")
        st.markdown("**📝 Your Question:**")
        st.info(st.session_state.transcribed_text)

        # Show edit mode if active
        if "edit_mode" in st.session_state and st.session_state.edit_mode:
            edited_text = st.text_area("Edit your question:", value=st.session_state.transcribed_text, height=100, key="edit_text_area")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Send", use_container_width=True, type="primary", key="send_edited"):
                    user_input = edited_text
                    del st.session_state.transcribed_text
                    del st.session_state.edit_mode
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_edit"):
                    del st.session_state.edit_mode
                    st.rerun()
            with col3:
                if st.button("🗑️ Discard", use_container_width=True, key="discard_edit"):
                    del st.session_state.transcribed_text
                    del st.session_state.edit_mode
                    st.session_state.voice_input_counter += 1
                    st.rerun()
        else:
            # Show Send, Edit, Discard buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Send Question", use_container_width=True, type="primary", key="send_transcribed"):
                    user_input = st.session_state.transcribed_text
                    # Clear the transcribed text after use
                    del st.session_state.transcribed_text
            with col2:
                if st.button("✏️ Edit", use_container_width=True, key="edit_transcribed"):
                    # Store in a text area for editing
                    st.session_state.edit_mode = True
                    st.rerun()
            with col3:
                if st.button("🗑️ Discard", use_container_width=True, key="discard_transcribed"):
                    del st.session_state.transcribed_text
                    st.session_state.voice_input_counter += 1
                    st.rerun()

    st.markdown("---")

    # Chat input
    if not user_input:
        user_input = st.chat_input("Ask your legal question...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append(("user", user_input))

        # Get response with intent detection and auto-classification
        with st.spinner("Analyzing question and finding answer..."):
            # Step 1: Check for harmful intent
            is_safe = detect_harmful_intent(user_input)

            if not is_safe:
                # Block harmful questions
                response = """🚫 **Safety Alert**

I cannot provide assistance with this request as it appears to involve potentially harmful or illegal activities.

**This legal advisory system is designed to:**
✅ Provide information about legal rights and protections
✅ Explain laws and legal procedures
✅ Help understand legal consequences
✅ Offer educational information about Indian law

**I cannot help with:**
❌ Planning or committing illegal activities
❌ Evading legal consequences
❌ Harming others or their rights
❌ Fraudulent or deceptive practices

If you have a legitimate legal question, please rephrase it. If you're facing a legal issue, I recommend consulting a licensed legal professional."""
                detected_category = "Safety Block"
            else:
                # Step 2: Auto-classify category
                detected_category, confidence = classify_category(user_input)

                if not detected_category:
                    response = "⚠️ Unable to classify your question. Please try asking a legal question related to Constitutional, Civil, Environmental, or Labor Law."
                    detected_category = "Unknown"
                else:
                    # Step 3: Get answer from detected category
                    response, detected_category = find_answer(user_input, detected_category)

                    # Prepend category badge to response
                    category_emoji = {
                        "Constitutional Law": "⚖️",
                        "Civil Law": "📜",
                        "Environmental Law": "🌿",
                        "Labor Law": "👷"
                    }
                    emoji = category_emoji.get(detected_category, "📚")
                    response = f"**{emoji} {detected_category}**\n\n{response}"

        # Add assistant response
        st.session_state.chat_history.append(("assistant", response))

        # Save to MongoDB
        save_chat_history(
            st.session_state.username,
            detected_category,
            user_input,
            response
        )

        # Clean up voice input session state
        if "transcribed_text" in st.session_state:
            del st.session_state.transcribed_text
        if "edit_mode" in st.session_state:
            del st.session_state.edit_mode

        # Increment counter to reset audio recorder for next question
        st.session_state.voice_input_counter += 1

        st.rerun()

        

# ---------------------------
# App Entry Point
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    show_auth_page()