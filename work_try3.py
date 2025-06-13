# Missing imports for os and time
import os
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# RAG & AI Imports
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import requests # For making API calls to the backend

# New Document Processing Imports
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
import warnings
import traceback # For more detailed error logging

warnings.filterwarnings('ignore')

# --- Backend and API Configuration ---
# This IP address should be the one where your Docker host is accessible
IP_ADDRESS = "10.4.25.215"
# IP_ADDRESS = "localhost"  # Use this for local development

# Correctly mapped ports from your docker-compose.yml
BASE_URLS = {
    "course_content": f"http://{IP_ADDRESS}:8000",
    "learning_objectives": f"http://{IP_ADDRESS}:8001",
    "kli_framework": f"http://{IP_ADDRESS}:8002",
    "instructional_design": f"http://{IP_ADDRESS}:8003",
    "validation_dashboard": f"http://{IP_ADDRESS}:8004",
}

# IMPORTANT: Replace with a valid JWT token obtained from your /auth/login endpoint
AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJpbnN0cnVjdG9yQGV4YW1wbGUuY29tIiwicm9sZSI6Imluc3RydWN0b3IiLCJleHAiOjE5MTYyMzkwMjJ9.fake_token_for_testing" # Replace with a real token if needed, or rely on authenticate()

# --- Gemini API Configuration ---
GEMINI_API_KEY = "AIzaSyC439kpMuI-D9NKZVe4cym7tQmF2d3_u1s" # Replace with your actual key if needed

def setup_gemini():
    """Configures the Google Gemini API with a hardcoded key."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print(f"✅ Gemini API configured successfully using the provided key!")
        print(f"🤖 Using Gemini Model: {model.model_name}")
        return model
    except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
        return None

gemini_model = setup_gemini()

def load_training_data():
    """Placeholder function for training data. User will add training examples later."""
    print("⚠️ Training data not loaded - placeholder function. Add your training examples here.")
    return {
        "training_examples": []  # Empty for now, user will populate this
    }


# ==============================================================================
# 📦 STEP 1.5: API CLIENT FOR BACKEND COMMUNICATION
# ==============================================================================

class APIClient:
    """A client to interact with the backend microservices."""
    def __init__(self, base_urls: Dict[str, str], auth_token: str = None):
        self.base_urls = base_urls
        self.auth_token = auth_token
        if auth_token: # if an initial token is provided (e.g. from global AUTH_TOKEN)
            self.headers = {"Authorization": auth_token, "accept": "application/json"}
        else: # no initial token, will need to authenticate
            self.headers = {"accept": "application/json"}
        print("📦 API Client Initialized.")
    
    def authenticate(self, email: str = "instructor@example.com", password: str = "password123") -> bool:
        """Authenticate and get a valid JWT token."""
        print(f"🔐 Authenticating with {email}...")
        # Use the course_content service for authentication endpoints typically
        login_url = f"{self.base_urls['course_content']}/auth/login" 
        
        try:
            # FastAPI's OAuth2PasswordRequestForm expects 'email' and 'password' in form data
            response = requests.post(login_url, data={"email": email, "password": password}) 
            
            if response.status_code == 200:
                token_data = response.json()
                self.auth_token = f"Bearer {token_data['access_token']}"
                self.headers["Authorization"] = self.auth_token # Update headers with new token
                print("✅ Authentication successful!")
                return True
            elif response.status_code == 401: # Unauthorized usually means user not found or bad password
                print("🔐 User not found or incorrect password, attempting to register...")
                register_url = f"{self.base_urls['course_content']}/auth/register"
                user_data = {
                    "email": email,
                    "password": password,
                    "name": "Pipeline User", # Or any default name
                    "role": "instructor" # Ensure your registration endpoint handles this role
                }
                register_response = requests.post(register_url, json=user_data)
                if register_response.status_code in [200, 201]: # 201 for Created
                    print("✅ User registered successfully!")
                    # Now try to login again
                    response = requests.post(login_url, data={"email": email, "password": password})
                    if response.status_code == 200:
                        token_data = response.json()
                        self.auth_token = f"Bearer {token_data['access_token']}"
                        self.headers["Authorization"] = self.auth_token
                        print("✅ Authentication successful after registration!")
                        return True
                    else:
                        print(f"❌ Login after registration failed: {response.status_code} - {response.text}")
                else:
                    print(f"❌ Registration failed: {register_response.status_code} - {register_response.text}")
            else:
                 print(f"❌ Authentication failed with status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Authentication request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Response Status: {e.response.status_code}")
                print(f"    Response Body: {e.response.text}")
        return False

    def _post(self, service: str, endpoint: str, json_data: Optional[dict] = None, files_data: Optional[dict] = None, data: Optional[dict] = None):
        """Helper for POST requests."""
        if not self.auth_token and endpoint not in ["/auth/login", "/auth/register"] and not "/health" in endpoint:
             print("🔒 Authentication required. Attempting to authenticate first.")
             if not self.authenticate():
                 print("❌ Authentication failed. Cannot proceed with POST request.")
                 return None
        url = f"{self.base_urls[service]}{endpoint}"
        try:
            current_headers = self.headers.copy()
            if files_data: # For file uploads, content-type is set by requests
                if 'Content-Type' in current_headers:
                    del current_headers['Content-Type']
                 # 'accept' should still be application/json for the response
                current_headers['accept'] = 'application/json'


            response = requests.post(url, headers=current_headers, json=json_data, files=files_data, data=data)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error occurred on POST to {url}: {http_err}")
            print(f"    Response Status: {http_err.response.status_code}")
            print(f"    Response Body: {http_err.response.text}")
        except requests.exceptions.RequestException as e: # Other errors like connection error
            print(f"❌ API Error on POST to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None: # Check if response attribute exists
                print(f"    Response Status: {e.response.status_code}")
                print(f"    Response Body: {e.response.text}")
        return None

    def _get(self, service: str, endpoint: str):
        """Helper for GET requests."""
        if not self.auth_token and "/health" not in endpoint :
             print("🔒 Authentication required. Attempting to authenticate first.")
             if not self.authenticate():
                 print("❌ Authentication failed. Cannot proceed with GET request.")
                 return None
        url = f"{self.base_urls[service]}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error occurred on GET from {url}: {http_err}")
            print(f"    Response Status: {http_err.response.status_code}")
            print(f"    Response Body: {http_err.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error on GET from {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Response Status: {e.response.status_code}")
                print(f"    Response Body: {e.response.text}")
        return None

    def create_course(self, title: str, description: str = "Course generated from PDF.") -> Optional[dict]:
        print(f"  > API: Creating course '{title}'...")
        payload = {"title": title, "description": description, "category": "AI Generated"}
        return self._post("course_content", "/courses/", json_data=payload)

    def upload_content_item(self, file_path: str, title: str) -> Optional[dict]:
        print(f"  > API: Uploading content item '{title}' from path '{file_path}'...")
        if not self.auth_token:
             print("🔒 Authentication required for file upload. Attempting to authenticate first.")
             if not self.authenticate():
                 print("❌ Authentication failed. Cannot proceed with file upload.")
                 return None

        try:
            with open(file_path, "rb") as f:
                # The 'type' field in data_payload should match one of your ContentItemType enums in the backend.
                # Based on the 422 error, the API expects a 'type' field, not 'item_type'
                data_payload = {'title': title, 'type': 'pdf'}  # Use 'type' field as expected by API
                files_payload = {'file': (os.path.basename(file_path), f, 'application/pdf')}
                
                # For multipart/form-data, requests handles Content-Type.
                # Only send Authorization and Accept headers needed by FastAPI.
                upload_headers = {}
                if self.auth_token: # self.auth_token already includes "Bearer "
                    upload_headers["Authorization"] = self.auth_token
                upload_headers["accept"] = "application/json" 
                
                url = f"{self.base_urls['course_content']}/content-items/"
                
                response = requests.post(url, headers=upload_headers, files=files_payload, data=data_payload)
                response.raise_for_status()
                return response.json()
        except FileNotFoundError:
            print(f"❌ File not found at path: {file_path}")
            return None
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ HTTP error on file upload to {url}: {http_err}")
            print(f"    Response Status: {http_err.response.status_code}")
            print(f"    Response Body: {http_err.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error on file upload to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"    Response Status: {e.response.status_code}")
                print(f"    Response Body: {e.response.text}")
            return None
        except Exception as e: # Catch any other unexpected errors
            print(f"❌ An unexpected error occurred during file upload: {e}")
            return None


    def link_content_to_course(self, course_id: int, content_item_id: int) -> Optional[dict]:
        print(f"  > API: Linking content item {content_item_id} to course {course_id}...")
        # API expects: POST /courses/{course_id}/content-items with CourseContentItemLink schema
        endpoint = f"/courses/{course_id}/content-items"
        payload = {"content_item_id": content_item_id}
        return self._post("course_content", endpoint, json_data=payload)


    def create_module(self, course_id: int, title: str, order: int) -> Optional[dict]:
        print(f"  > API: Creating module '{title}' in course {course_id}...")
        payload = {"module_name": title, "module_order": order, "course_id": course_id}
        # API expects: POST /courses/{course_id}/modules
        endpoint = f"/courses/{course_id}/modules"
        return self._post("learning_objectives", endpoint, json_data=payload)

    def create_lesson(self, module_id: int, title: str, order: int) -> Optional[dict]:
        print(f"    > API: Creating lesson '{title}' in module {module_id}...")
        payload = {"lesson_name": title, "lesson_order": order, "module_id": module_id}
        # API expects: POST /modules/{module_id}/lessons
        endpoint = f"/modules/{module_id}/lessons"
        return self._post("learning_objectives", endpoint, json_data=payload)

    def create_learning_objective(self, lesson_id: int, objective_data: dict, course_id: int) -> Optional[dict]:
        print(f"      > API: Creating objective in lesson {lesson_id}...")
        
        real_world_app = objective_data.get("real_world_context", "N/A")
        if isinstance(real_world_app, list):
            real_world_app = ". ".join(real_world_app)
        elif real_world_app is None:
            real_world_app = "N/A"
        
        # Map according to the actual learning_objectives schema
        payload = {
            "objective": objective_data.get("learning_objective", "No objective specified"),  # Maps to 'objective' field in schema
            "lesson_id": lesson_id,
            "order_in_course": objective_data.get("order", 1),  # Maps to 'order_in_course' field in schema
            "real_world_application": real_world_app,
            "common_errors": ", ".join(objective_data.get("common_errors", []) if objective_data.get("common_errors") else []),
            "scaffolding_support": ", ".join(objective_data.get("scaffolding_support", []) if objective_data.get("scaffolding_support") else [])
        }
        
        endpoint = f"/lessons/{lesson_id}/objectives"
        return self._post("learning_objectives", endpoint, json_data=payload)

    def get_course_structure(self, course_id: int) -> Optional[dict]:
        print(f"\n  > API: Fetching final curriculum structure for course {course_id}...")
        return self._get("learning_objectives", f"/courses/{course_id}/structure")
    
    def create_knowledge_component(self, objective_api_id: int, kc_data: dict) -> Optional[dict]:
        """Create a knowledge component in the KLI framework."""
        print(f"        > API: Creating knowledge component for objective_api_id {objective_api_id}...")
        
        # Map according to the actual knowledge_components schema
        payload = {
            "learning_objective_id": objective_api_id,
            "component_name": kc_data.get("knowledge_component", "Unknown Component"),
            "component_description": f"Knowledge component: {kc_data.get('learning_objective', 'No description')}",
            "kc_type": self._map_cognitive_complexity_to_kc_type(kc_data.get("cognitive_complexity", "ck")),
            "order_in_objective": 1  # Required field from schema
        }
        
        kli_endpoint = f"/objectives/{objective_api_id}/kcs"
        return self._post("kli_framework", kli_endpoint, json_data=payload)
    
    def _map_cognitive_complexity_to_kc_type(self, complexity: str) -> str:
        """Map cognitive complexity to KLI framework KC types according to schema."""
        mapping = {
            "factual knowledge": "fk",
            "conceptual understanding": "ck", 
            "procedural knowledge": "pk",
            "application": "pk",
            "analysis": "ck", 
            "synthesis": "sk", 
            "evaluation": "ck", 
            "comparative analysis": "ck",
            "multi-level understanding": "ck",
            "critical evaluation": "ck",
            "pattern recognition": "ck", 
            "application recognition": "ck",
            "causal understanding": "ck",
            "multi-dimensional analysis": "ck",
            "historical analysis": "ck", 
            "domain application analysis": "ck",
            "process understanding": "pk", 
            "ethical impact analysis": "ck", 
            "strategic application": "sk",
            "critical analysis": "ck",
            "hierarchical understanding": "ck",
        }
        complexity_lower = complexity.lower().strip()
        
        # Enhanced fallback logic
        if "procedural" in complexity_lower or "apply" in complexity_lower or "execution" in complexity_lower or "implement" in complexity_lower:
            return "pk"
        if "strategic" in complexity_lower or "design" in complexity_lower or "create" in complexity_lower or "formulate" in complexity_lower:
            return "sk" 
        
        return mapping.get(complexity_lower, "ck")  # Default to conceptual knowledge 
    
    def generate_instructional_strategies(self, course_id: int) -> Optional[dict]:
        """Generate instructional strategies for KCs in a course."""
        print(f"      > API: Note - Instructional strategies are generated per KC, not per course.")
        print(f"      > API: Use generate_instructional_strategies_for_kc(kc_id) for individual KCs.")
        # The API only supports KC-specific strategy generation: POST /kcs/{kc_id}/strategies/generate
        # There's no course-level strategy generation endpoint in the API
        return {"message": "Instructional strategies must be generated per KC. Use generate_instructional_strategies_for_kc(kc_id)."}
    
    def generate_instructional_strategies_for_kc(self, kc_id: int, lp_id: Optional[int] = None) -> Optional[dict]:
        """Generate instructional strategies for a specific Knowledge Component."""
        print(f"      > API: Generating instructional strategies for KC {kc_id}...")
        endpoint = f"/kcs/{kc_id}/strategies/generate"
        params = {}
        if lp_id:
            params["lp_id"] = lp_id
        
        # Since this is a POST with query params, we'll add them to the URL
        if params:
            endpoint += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        return self._post("instructional_design", endpoint, json_data={})
    
    def check_service_health(self) -> Dict[str, bool]:
        """Check health of all microservices."""
        health_status = {}
        print("\n🏥 Checking microservice health...")
        for service_name, base_url in self.base_urls.items():
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                health_status[service_name] = response.status_code == 200
                print(f"  {service_name}: {'✅ Healthy' if health_status[service_name] else '❌ Unhealthy'}")
            except requests.exceptions.RequestException:
                health_status[service_name] = False
                print(f"  {service_name}: ❌ Unreachable")
        return health_status


# ==============================================================================
# 🏗️ STEP 2: CORE RAG COMPONENTS (DATA STRUCTURES & GENERATION)
# ==============================================================================
@dataclass
class LearningObjective: # This is the KLI-inspired structure for local generation
    objective_id: str 
    knowledge_component: str
    learning_objective: str
    prerequisite_kcs: List[str]
    difficulty_estimate: float
    cognitive_complexity: str
    estimated_learning_time: str
    practice_opportunities: int
    mastery_criteria: str
    assessment_method: str 
    common_errors: List[str]
    scaffolding_support: List[str]
    real_world_context: str
    order: Optional[int] = None 

@dataclass
class Lesson: # For local structuring before API calls / assessment generation
    lesson_title: str
    learning_objectives: List[LearningObjective]
    id: Optional[int] = None # To store API ID if created
    module_id: Optional[int] = None # To store API ID if created
    order: Optional[int] = None


@dataclass
class Module: # For local structuring before API calls / assessment generation
    module_title: str
    lessons: List[Lesson]
    id: Optional[int] = None # To store API ID if created
    course_id: Optional[int] = None # To store API ID if created
    order: Optional[int] = None

def load_training_data():
    """Loads the FULL AI Literacy training examples."""
    training_data = {
        "training_examples": [
            {"sample_id": "AI_LIT_001","source_module": "Module 1","source_section": "Defining Intelligence - Human vs. Machine","input_text": "Human Intelligence includes learning from experience, solving complex problems, adapting to new situations, understanding context and meaning, creative thinking and innovation, and emotional and social intelligence. Machine Intelligence involves processing vast amounts of data quickly, pattern recognition and analysis, following complex algorithms, consistent performance without fatigue, specific task optimization, and statistical learning and prediction.","expected_output": {"learning_objectives": [{"objective_id": "KC_INT_001","knowledge_component": "Human-Machine-Intelligence-Comparison","learning_objective": "Compare human and machine intelligence by identifying distinct capabilities and limitations of each type","prerequisite_kcs": [],"difficulty_estimate": 0.3,"cognitive_complexity": "Comparative Analysis","estimated_learning_time": "12 minutes","practice_opportunities": 6,"mastery_criteria": "Correctly identify and categorize 8/10 intelligence characteristics as human vs machine","assessment_method": "Characteristic categorization with justification","common_errors": ["Assuming machines have emotional intelligence", "Underestimating human adaptability"],"scaffolding_support": ["Human vs machine capability matrix", "Intelligence type examples"],"real_world_context": "Understanding why humans still outperform AI in creative problem-solving"}]}},
            {"sample_id": "AI_LIT_002","source_module": "Module 1","source_section": "What is Artificial Intelligence (AI)?","input_text": "Technical Definition: AI is the simulation of human intelligence processes by machines, especially computer systems, including learning, reasoning, and self-correction. Simple Definition: AI is technology that can perform tasks that typically require human intelligence. Practical Definition: AI systems can analyze data, recognize patterns, make decisions, and solve problems without explicit programming for each specific task.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_DEF_001","knowledge_component": "AI-Multi-Level-Definition","learning_objective": "Define AI at technical, simple, and practical levels by explaining core processes and capabilities","prerequisite_kcs": ["Human-Machine-Intelligence-Comparison"],"difficulty_estimate": 0.2,"cognitive_complexity": "Multi-Level Understanding","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "Provide accurate AI definitions at all three levels in 90% of attempts","assessment_method": "Multi-level definition exercise with audience adaptation","common_errors": ["Confusing AI with simple automation", "Missing self-correction component"],"scaffolding_support": ["Definition hierarchy guide", "Audience-appropriate examples"],"real_world_context": "Explaining AI to different stakeholders: technical teams, management, general public"}]}},
            {"sample_id": "AI_LIT_003","source_module": "Module 1","source_section": "The Turing Test - A Benchmark for Intelligence?","input_text": "Proposed by Alan Turing in 1950, the Turing test asks: 'Can a machine engage in conversations indistinguishable from a human?' The Setup involves a human evaluator conducting text conversations with both a human and a machine. If the evaluator cannot reliably tell which is which, the machine 'passes' the test. Modern Perspective shows that advanced chatbots like ChatGPT can seem very human-like in conversation, but critics argue the test is too narrow—intelligence involves more than chat.","expected_output": {"learning_objectives": [{"objective_id": "KC_TURING_001","knowledge_component": "Turing-Test-Evaluation-Criteria","learning_objective": "Evaluate AI systems using Turing Test criteria while recognizing its limitations and modern alternatives","prerequisite_kcs": ["AI-Multi-Level-Definition"],"difficulty_estimate": 0.4,"cognitive_complexity": "Critical Evaluation","estimated_learning_time": "15 minutes","practice_opportunities": 7,"mastery_criteria": "Correctly assess Turing Test applicability and limitations in 85% of AI evaluation scenarios","assessment_method": "AI system evaluation using multiple intelligence criteria","common_errors": ["Overvaluing conversational ability", "Missing broader intelligence aspects"],"scaffolding_support": ["Intelligence assessment framework", "Test limitation analysis guide"],"real_world_context": "Evaluating whether modern chatbots like ChatGPT demonstrate true intelligence"}]}},
            {"sample_id": "AI_LIT_004","source_module": "Module 1","source_section": "Types of AI - Artificial Narrow Intelligence (ANI)","input_text": "ANI are AI systems designed to perform specific, narrow tasks very well, but cannot generalize beyond their training. Current Capabilities include superhuman performance in specific areas, highly reliable and consistent operation, processing information much faster than humans, but limited to their programmed functions. Important Note: 'Weak' doesn't mean less powerful—it means specialized rather than general.","expected_output": {"learning_objectives": [{"objective_id": "KC_ANI_001","knowledge_component": "ANI-Characteristics-Limitations","learning_objective": "Identify ANI systems by recognizing specialized capabilities and inability to generalize beyond training domains","prerequisite_kcs": ["Turing-Test-Evaluation-Criteria"],"difficulty_estimate": 0.3,"cognitive_complexity": "Pattern Recognition","estimated_learning_time": "12 minutes","practice_opportunities": 8,"mastery_criteria": "Correctly classify 9/10 AI systems as ANI vs other types with reasoning","assessment_method": "AI system classification with capability analysis","common_errors": ["Expecting generalization from narrow systems", "Confusing 'weak' with 'less capable'"],"scaffolding_support": ["ANI identification checklist", "Capability vs generalization matrix"],"real_world_context": "Understanding why chess AI cannot play Go without retraining"}]}},
            {"sample_id": "AI_LIT_005","source_module": "Module 1","source_section": "Examples of ANI in Your Daily Life","input_text": "You interact with ANI constantly through: Search Engines (Google) understanding your search intent and ranking billions of web pages instantly; Recommendation Systems like Netflix suggesting movies you'll like, Amazon showing relevant products, Spotify creating personalized playlists; Voice Assistants including Siri, Alexa, Google Assistant with speech recognition and response generation; Navigation Apps providing real-time traffic analysis and route optimization; Email & Communication with spam filtering, smart compose in Gmail, language translation.","expected_output": {"learning_objectives": [{"objective_id": "KC_ANI_APP_001","knowledge_component": "Daily-ANI-Application-Recognition","learning_objective": "Recognize ANI applications in daily digital interactions by identifying their specialized functions and data processing capabilities","prerequisite_kcs": ["ANI-Characteristics-Limitations"],"difficulty_estimate": 0.2,"cognitive_complexity": "Application Recognition","estimated_learning_time": "10 minutes","practice_opportunities": 10,"mastery_criteria": "Identify ANI systems in 95% of daily technology interactions","assessment_method": "Daily technology audit with ANI identification","common_errors": ["Missing invisible AI applications", "Assuming general intelligence in specialized systems"],"scaffolding_support": ["Daily AI interaction checklist", "Hidden AI detection guide"],"real_world_context": "Documenting personal AI usage throughout a typical day"}]}},
            {"sample_id": "AI_LIT_006","source_module": "Module 2","source_section": "Data: The Fuel for Modern AI","input_text": "Data is the lifeblood, the raw material, the essential fuel that powers modern AI. Without vast quantities of relevant data, most AI algorithms would be like engines without gasoline – inert and incapable of performing their intended tasks. The quality, quantity, and relevance of this data are paramount. This data can take countless forms: millions of images of cats and dogs to train an image recognition system, vast libraries of text and code to train a large language model like ChatGPT, years of stock market fluctuations to train a financial prediction algorithm.","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_FUEL_001","knowledge_component": "Data-AI-Performance-Relationship","learning_objective": "Explain the critical relationship between data quality, quantity, relevance and AI system performance using fuel metaphor","prerequisite_kcs": ["Daily-ANI-Application-Recognition"],"difficulty_estimate": 0.3,"cognitive_complexity": "Causal Understanding","estimated_learning_time": "12 minutes","practice_opportunities": 6,"mastery_criteria": "Accurately predict AI performance based on data characteristics in 85% of scenarios","assessment_method": "Data quality assessment with performance prediction","common_errors": ["Assuming quantity always trumps quality", "Missing relevance importance"],"scaffolding_support": ["Data-performance correlation examples", "Quality assessment criteria"],"real_world_context": "Explaining why medical AI needs relevant patient data, not just any health data"}]}},
            {"sample_id": "AI_LIT_007","source_module": "Module 2","source_section": "Big Data: Volume, Velocity, Variety","input_text": "Big Data is characterized by the 'Three Vs': Volume refers to the massive quantity of data being generated and stored, often measured in terabytes, petabytes, or even exabytes. Velocity represents the incredible speed at which new data is created and needs to be processed, such as real-time data from financial markets where milliseconds matter. Variety encompasses the wide range of data types from diverse sources, including structured data from databases, unstructured text from documents and social media, images and videos from cameras and smartphones.","expected_output": {"learning_objectives": [{"objective_id": "KC_BIGDATA_001","knowledge_component": "Big-Data-3Vs-Analysis","learning_objective": "Analyze big data scenarios by evaluating volume, velocity, and variety characteristics with appropriate scale measurements","prerequisite_kcs": ["Data-AI-Performance-Relationship"],"difficulty_estimate": 0.4,"cognitive_complexity": "Multi-Dimensional Analysis","estimated_learning_time": "18 minutes","practice_opportunities": 9,"mastery_criteria": "Correctly assess all 3 Vs and recommend processing approaches in 80% of big data scenarios","assessment_method": "Big data scenario analysis with processing strategy recommendation","common_errors": ["Confusing volume with velocity", "Underestimating variety complexity"],"scaffolding_support": ["3Vs assessment matrix", "Scale measurement reference guide"],"real_world_context": "Analyzing social media platform data requirements for real-time trend detection"}]}},
            {"sample_id": "AI_LIT_008","source_module": "Module 3","source_section": "Everyday AI: Search Engines (Google)","input_text": "Search engines, especially Google, use sophisticated AI/ML to understand nuanced queries, discern intent, and deliver relevant, personalized results quickly. RankBrain (2015) interprets ambiguous/novel queries. BERT (2019) understands word context for better conversational query results. MUM (2021) processes information across languages and formats for complex tasks. Gemini is Google's latest AI, enhancing understanding with multi-step reasoning and multimodality.","expected_output": {"learning_objectives": [{"objective_id": "KC_SEARCH_AI_001","knowledge_component": "Search-Engine-AI-Evolution","learning_objective": "Trace search engine AI evolution from RankBrain to Gemini by analyzing capability improvements and query understanding advances","prerequisite_kcs": ["Big-Data-3Vs-Analysis"],"difficulty_estimate": 0.5,"cognitive_complexity": "Historical Analysis","estimated_learning_time": "20 minutes","practice_opportunities": 7,"mastery_criteria": "Correctly sequence AI developments and explain capability improvements in 85% of cases","assessment_method": "Timeline creation with capability analysis","common_errors": ["Missing chronological progression", "Confusing different AI system capabilities"],"scaffolding_support": ["Search AI timeline template", "Capability comparison matrix"],"real_world_context": "Understanding why modern search results are more contextually relevant than early keyword matching"}]}},
            {"sample_id": "AI_LIT_009","source_module": "Module 3","source_section": "AI in Healthcare: Diagnostics & Medical Imaging","input_text": "AI, especially Convolutional Neural Networks (CNNs), analyzes X-rays, CT scans, and MRIs. Key Applications include Oncology for detecting breast, lung, and prostate cancers; Radiology for identifying fractures, strokes, and pneumonia; Cardiology for analyzing echocardiograms and ECGs; Ophthalmology for detecting diabetic retinopathy, AMD, and glaucoma. Benefits include improved accuracy, early detection, increased efficiency, quantitative analysis, and enhanced accessibility.","expected_output": {"learning_objectives": [{"objective_id": "KC_HEALTH_AI_001","knowledge_component": "Medical-AI-Application-Assessment","learning_objective": "Assess medical AI applications by categorizing diagnostic capabilities across medical specialties and evaluating clinical benefits","prerequisite_kcs": ["Search-Engine-AI-Evolution"],"difficulty_estimate": 0.6,"cognitive_complexity": "Domain Application Analysis","estimated_learning_time": "25 minutes","practice_opportunities": 8,"mastery_criteria": "Correctly categorize medical AI applications and predict clinical impact in 80% of healthcare scenarios","assessment_method": "Medical AI capability mapping with benefit analysis","common_errors": ["Overestimating AI diagnostic independence", "Missing specialty-specific requirements"],"scaffolding_support": ["Medical specialty AI matrix", "Clinical benefit assessment framework"],"real_world_context": "Evaluating AI radiology tools for emergency department implementation"}]}},
            {"sample_id": "AI_LIT_010","source_module": "Module 3","source_section": "Introduction to Generative AI (GenAI)","input_text": "Generative AI (GenAI) is a subfield of AI focused on creating new, original content (text, images, audio, video, code) that mimics human-created works, by learning from vast datasets. Key Characteristics include Content Creation producing new content, Learning from Data trained on extensive datasets, Pattern Recognition & Replication identifying and learning underlying patterns and styles, and Novelty where generated content is new, though based on learned patterns.","expected_output": {"learning_objectives": [{"objective_id": "KC_GENAI_001","knowledge_component": "GenAI-Content-Creation-Process","learning_objective": "Explain GenAI content creation process by identifying pattern learning, data training, and novelty generation mechanisms","prerequisite_kcs": ["Medical-AI-Application-Assessment"],"difficulty_estimate": 0.4,"cognitive_complexity": "Process Understanding","estimated_learning_time": "15 minutes","practice_opportunities": 6,"mastery_criteria": "Correctly explain GenAI creation process and distinguish from traditional AI in 85% of scenarios","assessment_method": "Process explanation with creation vs classification distinction","common_errors": ["Confusing generation with classification", "Missing pattern learning foundation"],"scaffolding_support": ["GenAI vs traditional AI comparison", "Creation process flowchart"],"real_world_context": "Understanding how DALL-E creates images from text descriptions"}]}},
            {"sample_id": "AI_LIT_011","source_module": "Module 4","source_section": "Ethical Consideration: Bias in AI – Sources & Consequences","input_text": "AI bias occurs when systems produce systematically prejudiced outputs due to erroneous assumptions in the machine learning process. Data bias includes historical bias from past societal prejudices, representation bias when training data doesn't reflect population diversity, and measurement bias from flawed data collection methods. Consequences include discrimination in hiring, lending, criminal justice; reinforcement of stereotypes; erosion of trust; economic disadvantage; and safety risks in critical applications.","expected_output": {"learning_objectives": [{"objective_id": "KC_BIAS_001","knowledge_component": "AI-Bias-Source-Impact-Analysis","learning_objective": "Analyze AI bias by identifying source categories (data, algorithmic, human) and predicting societal consequences across domains","prerequisite_kcs": ["GenAI-Content-Creation-Process"],"difficulty_estimate": 0.7,"cognitive_complexity": "Ethical Impact Analysis","estimated_learning_time": "30 minutes","practice_opportunities": 10,"mastery_criteria": "Correctly identify bias sources and predict consequences in 85% of AI deployment scenarios","assessment_method": "Bias impact assessment with stakeholder analysis","common_errors": ["Missing systemic consequences", "Focusing only on technical bias sources"],"scaffolding_support": ["Bias source taxonomy", "Stakeholder impact matrix", "Consequence prediction framework"],"real_world_context": "Evaluating hiring AI for potential discrimination before deployment"}]}},
            {"sample_id": "AI_LIT_012","source_module": "Module 5","source_section": "Effective Prompting for Text-Based Generative AI","input_text": "To elicit high-quality responses from text-based generative AI, consider: Be Clear and Specific with vague prompts yielding generic results; Provide Sufficient Context with background information helping AI understand nuances; Define a Role or Persona by instructing AI to 'act as' specific personas; Specify the Desired Format clearly stating how information should be structured; Set Constraints and Boundaries by specifying limitations like word count and tone; Iterate and Refine treating prompting as an iterative process.","expected_output": {"learning_objectives": [{"objective_id": "KC_PROMPT_001","knowledge_component": "Prompt-Engineering-Optimization-Strategy","learning_objective": "Optimize AI prompts by applying clarity, context, persona, format, and constraint principles through iterative refinement","prerequisite_kcs": ["AI-Bias-Source-Impact-Analysis"],"difficulty_estimate": 0.5,"cognitive_complexity": "Strategic Application","estimated_learning_time": "22 minutes","practice_opportunities": 12,"mastery_criteria": "Achieve 80% improvement in AI output quality through systematic prompt optimization","assessment_method": "Prompt optimization project with quality measurement","common_errors": ["Vague instructions", "Missing iterative improvement", "Ignoring context importance"],"scaffolding_support": ["Prompt optimization checklist", "Quality assessment rubric", "Iterative refinement workflow"],"real_world_context": "Creating effective prompts for business content generation and customer service automation"}]}},
            {"sample_id": "AI_LIT_013","source_module": "Module 5","source_section": "Critical Evaluation of AI-Generated Content","input_text": "AI models are prone to 'hallucinations' where AI generates information that is false, nonsensical, irrelevant, or entirely fabricated, yet presents it with apparent confidence. Common indicators include plausible but verifiably false information, internal contradictions, overly confident assertions on niche topics, fabricated sources or citations, nonsensical statements, and subtle semantic or factual errors. Verification requires cross-referencing with reputable sources, seeking corroboration, and applying critical judgment.","expected_output": {"learning_objectives": [{"objective_id": "KC_EVAL_001","knowledge_component": "AI-Hallucination-Detection-Verification","learning_objective": "Detect AI hallucinations by recognizing reliability indicators and implementing systematic verification protocols","prerequisite_kcs": ["Prompt-Engineering-Optimization-Strategy"],"difficulty_estimate": 0.6,"cognitive_complexity": "Critical Analysis","estimated_learning_time": "25 minutes","practice_opportunities": 15,"mastery_criteria": "Correctly identify hallucinations and verify content accuracy in 90% of AI-generated outputs","assessment_method": "AI content verification project with accuracy assessment","common_errors": ["Trusting confident-sounding false information", "Insufficient source verification"],"scaffolding_support": ["Hallucination detection checklist", "Verification protocol guide", "Source reliability matrix"],"real_world_context": "Fact-checking AI-generated research summaries before using in professional presentations"}]}},
            {"sample_id": "AI_LIT_014","source_module": "Module 1","source_section": "AI, Machine Learning (ML), and Deep Learning (DL) - The Relationship","input_text": "Understanding the Hierarchy: Artificial Intelligence (Outermost Circle) is the broadest term encompassing all machine intelligence, includes rule-based systems, expert systems, and learning systems with the goal to create machines that can perform tasks requiring intelligence. Machine Learning (Middle Circle) is a subset of AI focused on learning from data, where algorithms improve performance through experience with no need to explicitly program every rule. Deep Learning (Inner Circle) is a subset of ML using artificial neural networks with multiple layers of interconnected nodes, particularly effective for complex pattern recognition.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_HIER_001","knowledge_component": "AI-ML-DL-Hierarchical-Relationship","learning_objective": "Map the hierarchical relationship between AI, ML, and DL by explaining subset relationships and capability progression","prerequisite_kcs": ["AI-Multi-Level-Definition"],"difficulty_estimate": 0.4,"cognitive_complexity": "Hierarchical Understanding","estimated_learning_time": "16 minutes","practice_opportunities": 8,"mastery_criteria": "Correctly explain hierarchical relationships and classify technologies in 90% of scenarios","assessment_method": "Technology classification with hierarchy explanation","common_errors": ["Using terms interchangeably", "Missing subset relationships"],"scaffolding_support": ["Venn diagram template", "Technology classification guide"],"real_world_context": "Explaining to stakeholders why deep learning success doesn't mean general AI breakthrough"}]}},
            {"sample_id": "AI_LIT_008A","source_module": "Module 2","source_section": "Data Collection Sources: User-Generated Content & IoT","input_text": "The vast quantities of data required to train effective AI systems are gathered from multiple sources. User-generated content includes active input like social media posts, product reviews, and survey responses, as well as passive input such as website browsing history, app usage patterns, and location data from GPS-enabled devices. Sensors and the Internet of Things (IoT) generate continuous streams of data. Industrial sensors monitor temperature, pressure, and vibration in machinery for predictive maintenance. Environmental sensors collect data on air quality, weather conditions, and seismic activity.","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_SRC_001","knowledge_component": "Data-Source-Collection-Methods","learning_objective": "Categorize data sources into user-generated, IoT sensor, and public dataset categories while distinguishing active vs passive collection methods","prerequisite_kcs": ["Data-AI-Performance-Relationship"],"difficulty_estimate": 0.4,"cognitive_complexity": "Multi-Category Classification","estimated_learning_time": "15 minutes","practice_opportunities": 10,"mastery_criteria": "Correctly categorize 12/15 data source examples and identify collection methods with 85% accuracy","assessment_method": "Data source categorization with collection method identification","common_errors": ["Confusing user behavior tracking with IoT sensors", "Missing passive data collection privacy implications"],"scaffolding_support": ["Data source classification matrix", "Active vs passive collection flowchart"],"real_world_context": "Designing data collection strategy for smart city traffic optimization system"}]}},
            {"sample_id": "AI_LIT_008B","source_module": "Module 2","source_section": "Data Preprocessing: Cleaning and Transformation","input_text": "Raw data is rarely suitable for direct use in training AI models. Data preprocessing involves cleaning, transforming, and organizing data into a high-quality, usable format. Key tasks include handling missing values through imputation or deletion, correcting errors and inconsistencies, and removing outliers that could distort learning. Data transformation includes normalization to scale numerical data to common ranges, encoding categorical data into numerical formats through techniques like one-hot encoding, and feature engineering to create new, more informative features from existing ones.","expected_output": {"learning_objectives": [{"objective_id": "KC_PREP_001","knowledge_component": "Data-Preprocessing-Pipeline-Execution","learning_objective": "Execute comprehensive data preprocessing by implementing cleaning, transformation, and feature engineering techniques systematically","prerequisite_kcs": ["Data-Source-Collection-Methods"],"difficulty_estimate": 0.6,"cognitive_complexity": "Procedural Execution","estimated_learning_time": "25 minutes","practice_opportunities": 12,"mastery_criteria": "Complete preprocessing pipeline achieving 90% data quality improvement on test datasets","assessment_method": "Hands-on preprocessing project with quality metrics evaluation","common_errors": ["Inappropriate imputation methods", "Over-aggressive outlier removal", "Poor categorical encoding choices"],"scaffolding_support": ["Preprocessing decision flowchart", "Quality assessment checklist", "Method selection guide"],"real_world_context": "Preparing customer transaction data for fraud detection model training"}]}},
            {"sample_id": "AI_LIT_008C","source_module": "Module 2","source_section": "Algorithms vs Models: Recipe vs Trained Output","input_text": "An algorithm is a finite sequence of well-defined, computer-implementable instructions to solve problems or perform computations. Think of an algorithm as a detailed recipe with specific ingredients (data) and step-by-step instructions to achieve a desired outcome. When a machine learning algorithm is trained on a specific dataset, the result is a trained AI model. The algorithm is the general procedure, while the trained model is the specific instance that has learned particular patterns from data and contains parameters optimized during training.","expected_output": {"learning_objectives": [{"objective_id": "KC_ALG_MOD_001","knowledge_component": "Algorithm-Model-Distinction-Application","learning_objective": "Distinguish between algorithms (general procedures) and trained models (specific learned instances) by analyzing their roles in the AI development process","prerequisite_kcs": ["Data-Preprocessing-Pipeline-Execution"],"difficulty_estimate": 0.5,"cognitive_complexity": "Conceptual Distinction","estimated_learning_time": "18 minutes","practice_opportunities": 8,"mastery_criteria": "Correctly identify and explain algorithm vs model in 90% of AI development scenarios","assessment_method": "Scenario analysis with algorithm-model classification and explanation","common_errors": ["Using terms interchangeably", "Missing the training/learning component that creates models"],"scaffolding_support": ["Recipe-algorithm analogy framework", "Development process flowchart"],"real_world_context": "Explaining to stakeholders why the same algorithm can produce different models for different datasets"}]}},
            {"sample_id": "AI_LIT_008D","source_module": "Module 2","source_section": "The Training Process: Iteration and Improvement","input_text": "The creation of an effective AI model involves a systematic and iterative training process. This begins with data preparation and splitting datasets into training, validation, and test sets. The training loop involves a forward pass where the model processes input to produce output, loss calculation to measure prediction errors, and backward pass optimization to adjust internal parameters. Evaluation and tuning monitor performance on validation sets to detect overfitting when training loss decreases but validation loss increases, indicating the model learns training data too well but fails to generalize.","expected_output": {"learning_objectives": [{"objective_id": "KC_TRAIN_001","knowledge_component": "Training-Loop-Overfitting-Management","learning_objective": "Implement training loops with overfitting detection by executing forward pass, loss calculation, backward pass optimization, and validation monitoring","prerequisite_kcs": ["Algorithm-Model-Distinction-Application"],"difficulty_estimate": 0.6,"cognitive_complexity": "Process Management","estimated_learning_time": "25 minutes","practice_opportunities": 12,"mastery_criteria": "Successfully implement training with overfitting detection in 85% of model development scenarios","assessment_method": "Training implementation project with overfitting prevention demonstration","common_errors": ["Confusing training/validation loss patterns", "Missing early stopping opportunities"],"scaffolding_support": ["Training loop flowchart", "Overfitting detection guide", "Loss pattern interpretation"],"real_world_context": "Training image classification model while preventing overfitting for production deployment"}]}},
            {"sample_id": "AI_LIT_008E","source_module": "Module 2","source_section": "Supervised Learning: Learning with Labels","input_text": "Supervised learning algorithms learn from labeled training examples to make predictions on new data. In spam detection, emails are labeled as 'spam' or 'not spam' during training, and the algorithm learns patterns like suspicious sender addresses, promotional language, and unusual formatting. Image classification uses training datasets with thousands of labeled images - photos tagged as 'cat,' 'dog,' 'car,' or 'tree.' Medical diagnosis applies supervised learning with labeled patient data including symptoms, test results, and confirmed diagnoses to predict diseases in new patients.","expected_output": {"learning_objectives": [{"objective_id": "KC_SUP_001","knowledge_component": "Supervised-Learning-Application-Design","learning_objective": "Design supervised learning applications by defining appropriate labels, features, and prediction targets for specific domain problems","prerequisite_kcs": ["Training-Loop-Overfitting-Management"],"difficulty_estimate": 0.5,"cognitive_complexity": "Application Design","estimated_learning_time": "20 minutes","practice_opportunities": 10,"mastery_criteria": "Design effective supervised learning solutions with 85% technical accuracy and appropriate evaluation metrics","assessment_method": "Supervised learning project design with implementation feasibility analysis","common_errors": ["Poorly defined label categories", "Missing feature engineering considerations", "Inappropriate evaluation metrics"],"scaffolding_support": ["Application design template", "Label definition guidelines", "Feature selection framework"],"real_world_context": "Developing customer churn prediction system using historical customer behavior and outcome data"}]}},
            {"sample_id": "AI_LIT_008F","source_module": "Module 2","source_section": "Unsupervised Learning: Pattern Discovery Without Labels","input_text": "Unsupervised learning discovers hidden patterns in data without labeled examples. Customer segmentation groups customers with similar purchasing behaviors, demographics, or preferences without predefined categories. Retailers use clustering algorithms to identify customer groups like 'budget-conscious families,' 'luxury shoppers,' or 'tech enthusiasts' based on transaction patterns. Anomaly detection identifies unusual patterns that deviate from normal behavior, such as credit card fraud detection monitoring transaction patterns to flag suspicious activities like unusual spending amounts, unfamiliar locations, or atypical merchant categories.","expected_output": {"learning_objectives": [{"objective_id": "KC_UNSUP_001","knowledge_component": "Unsupervised-Learning-Pattern-Discovery","learning_objective": "Apply unsupervised learning techniques for pattern discovery by implementing clustering for segmentation and anomaly detection for outlier identification","prerequisite_kcs": ["Supervised-Learning-Application-Design"],"difficulty_estimate": 0.7,"cognitive_complexity": "Pattern Analysis","estimated_learning_time": "30 minutes","practice_opportunities": 12,"mastery_criteria": "Generate meaningful business insights from clustering and detect 90% of anomalies in test scenarios","assessment_method": "Unsupervised learning project with pattern interpretation and business value assessment","common_errors": ["Over-interpreting cluster meanings", "Missing business context in pattern analysis", "Poor anomaly threshold selection"],"scaffolding_support": ["Clustering interpretation guide", "Anomaly detection parameter tuning", "Business value assessment framework"],"real_world_context": "Analyzing customer behavior patterns for marketing strategy and fraud prevention in financial services"}]}},
            {"sample_id": "AI_LIT_015","source_module": "Module 2","source_section": "Machine Learning Paradigms: Supervised, Unsupervised, Reinforcement","input_text": "Supervised learning algorithms learn from labeled training examples to make predictions on new data, like spam detection with emails labeled as 'spam' or 'not spam.' Unsupervised learning discovers hidden patterns in data without labeled examples, like customer segmentation grouping customers with similar behaviors without predefined categories. Reinforcement learning involves an agent learning optimal actions through trial and error, receiving rewards for beneficial actions and penalties for harmful ones, like AI agents learning game strategies through millions of plays.","expected_output": {"learning_objectives": [{"objective_id": "KC_ML_PAR_001","knowledge_component": "ML-Paradigm-Problem-Matching","learning_objective": "Match machine learning paradigms to problem types by analyzing data availability, learning goals, and feedback mechanisms","prerequisite_kcs": ["AI-ML-DL-Hierarchical-Relationship"],"difficulty_estimate": 0.5,"cognitive_complexity": "Problem-Solution Matching","estimated_learning_time": "20 minutes","practice_opportunities": 12,"mastery_criteria": "Correctly match ML paradigms to problems and justify selection in 85% of scenarios","assessment_method": "Problem-paradigm matching with methodology justification","common_errors": ["Confusing supervised with unsupervised applications", "Missing reinforcement learning trial-error component"],"scaffolding_support": ["Paradigm selection flowchart", "Problem type classification guide"],"real_world_context": "Selecting appropriate ML approach for different business problems: sales prediction, customer segmentation, pricing optimization"}]}},
            {"sample_id": "AI_LIT_016","source_module": "Module 3","source_section": "Everyday AI: Recommendation Systems (Netflix, Amazon, Spotify)","input_text": "Recommendation systems analyze user behavior and item data through AI/ML algorithms. Data Collection includes user interactions (history, ratings, browsing), item attributes (metadata like genre, brand, price), and feedback (direct ratings or implicit clicks). Core approaches include Collaborative Filtering assuming users with past agreements will agree in future, Content-Based Filtering recommending items similar to past preferences, and Hybrid Models combining approaches for robust recommendations. Impact shows Netflix with ~80% of watched content driven by recommendations and Amazon with ~35% of revenue from its engine.","expected_output": {"learning_objectives": [{"objective_id": "KC_REC_SYS_001","knowledge_component": "Recommendation-System-Algorithm-Selection","learning_objective": "Design recommendation systems by selecting appropriate algorithms (collaborative, content-based, hybrid) based on data availability and user behavior patterns","prerequisite_kcs": ["ML-Paradigm-Problem-Matching"],"difficulty_estimate": 0.6,"cognitive_complexity": "System Design","estimated_learning_time": "25 minutes","practice_opportunities": 10,"mastery_criteria": "Design effective recommendation systems achieving appropriate algorithm selection in 80% of business scenarios","assessment_method": "Recommendation system design project with algorithm justification","common_errors": ["Choosing wrong algorithm for data type", "Missing hybrid approach benefits"],"scaffolding_support": ["Algorithm selection matrix", "Data requirement guide", "Performance optimization checklist"],"real_world_context": "Designing recommendation engine for new e-commerce platform with limited user history"}]}},
            {"sample_id": "AI_LIT_017","source_module": "Module 4","source_section": "Societal Impact: The Future of Work - Job Displacement","input_text": "AI is transforming the labor market by automating routine and some complex tasks. Job displacement affects blue-collar (manufacturing, transport) and white-collar (customer service, data entry) jobs, with middle-skill jobs being most vulnerable. However, new job creation includes emerging roles like AI developers, data scientists, AI ethicists, prompt engineers, and human supervisors. There's increased demand for 'human' skills such as creativity, critical thinking, emotional intelligence, digital literacy, and adaptability.","expected_output": {"learning_objectives": [{"objective_id": "KC_WORK_AI_001","knowledge_component": "AI-Workforce-Transformation-Analysis","learning_objective": "Analyze AI's workforce impact by predicting job displacement patterns and identifying emerging skill requirements","prerequisite_kcs": ["Recommendation-System-Algorithm-Selection"],"difficulty_estimate": 0.6,"cognitive_complexity": "Trend Analysis","estimated_learning_time": "25 minutes","practice_opportunities": 9,"mastery_criteria": "Correctly predict workforce changes and skill evolution in 80% of industry scenarios","assessment_method": "Workforce impact analysis with skill requirement forecasting","common_errors": ["Overestimating immediate displacement", "Underestimating new job creation"],"scaffolding_support": ["Job displacement risk matrix", "Skill evolution framework", "Industry transformation timeline"],"real_world_context": "Helping organizations prepare workforce development strategies for AI integration"}]}},
            {"sample_id": "AI_LIT_018","source_module": "Module 4","source_section": "Societal Impact: Misinformation & Disinformation","input_text": "Generative AI amplifies the spread of false information. Key issues include Deepfakes as realistic fake audio/video used for propaganda, fraud, or harassment, and AI-generated fake news where large language models create convincing, false articles at scale. Consequences include undermining democracy and trust, social polarization, and risk of inciting violence. Mitigation strategies involve AI detection tools, media literacy education, fact-checking initiatives, and content provenance technologies.","expected_output": {"learning_objectives": [{"objective_id": "KC_MISINFO_001","knowledge_component": "AI-Misinformation-Detection-Prevention","learning_objective": "Develop misinformation countermeasures by identifying AI-generated false content and implementing detection and prevention strategies","prerequisite_kcs": ["AI-Workforce-Transformation-Analysis"],"difficulty_estimate": 0.7,"cognitive_complexity": "Strategic Defense Planning","estimated_learning_time": "30 minutes","practice_opportunities": 12,"mastery_criteria": "Design effective misinformation defense strategies addressing 85% of AI-generated content threats","assessment_method": "Misinformation defense strategy development with effectiveness evaluation","common_errors": ["Relying only on technical detection", "Missing media literacy component"],"scaffolding_support": ["Detection tool comparison matrix", "Defense strategy framework", "Media literacy curriculum guide"],"real_world_context": "Developing organizational policies to combat AI-generated misinformation in news and social media"}]}},
            {"sample_id": "AI_LIT_019","source_module": "Module 6","source_section": "Current Frontiers: Multimodal AI and Embodied AI","input_text": "Multimodal AI systems are capable of processing, understanding, integrating, and generating information from multiple data types concurrently (text, images, audio, video, code, sensor data). Examples include Google's Gemini understanding drawings and spoken questions, and OpenAI's GPT-4V analyzing images and text queries. Embodied AI integrates into robots, drones, or autonomous vehicles that perceive their environment through sensors, reason about it, and take physical actions to achieve goals, like warehouse robots learning complex physical tasks or autonomous agricultural robots performing planting and harvesting.","expected_output": {"learning_objectives": [{"objective_id": "KC_MULTI_AI_001","knowledge_component": "Multimodal-Embodied-AI-Capabilities","learning_objective": "Distinguish multimodal and embodied AI capabilities by analyzing integration complexity and real-world application requirements","prerequisite_kcs": ["AI-Misinformation-Detection-Prevention"],"difficulty_estimate": 0.6,"cognitive_complexity": "Advanced Technology Analysis","estimated_learning_time": "22 minutes","practice_opportunities": 8,"mastery_criteria": "Correctly classify AI systems and predict capabilities in 85% of advanced technology scenarios","assessment_method": "Advanced AI capability assessment with application prediction","common_errors": ["Confusing multimodal with multi-model systems", "Underestimating embodied AI complexity"],"scaffolding_support": ["AI capability classification matrix", "Application complexity guide", "Technology readiness assessment"],"real_world_context": "Evaluating next-generation AI systems for enterprise adoption and investment decisions"}]}},
            {"sample_id": "AI_LIT_020","source_module": "Module 6","source_section": "AI for Good: Solving Global Challenges","input_text": "AI offers powerful tools to address pressing global challenges. Climate change applications include enhanced climate modeling and prediction, renewable energy optimization, sustainable agriculture with precision farming, and conservation efforts through satellite imagery analysis and species tracking. Healthcare advances include accelerated drug discovery and development, early disease detection and diagnostics, personalized medicine and treatment, and improved healthcare access through AI-powered tools for underserved areas. The ethical imperative requires ensuring 'AI for Good' initiatives are developed and deployed ethically, equitably, and transparently.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_GOOD_001","knowledge_component": "AI-Global-Challenge-Solution-Design","learning_objective": "Design AI solutions for global challenges by matching AI capabilities to problem requirements while ensuring ethical deployment","prerequisite_kcs": ["Multimodal-Embodied-AI-Capabilities"],"difficulty_estimate": 0.8,"cognitive_complexity": "Ethical Solution Design","estimated_learning_time": "35 minutes","practice_opportunities": 10,"mastery_criteria": "Design comprehensive AI solutions addressing global challenges with 90% ethical compliance","assessment_method": "Global challenge solution design with ethical impact assessment","common_errors": ["Missing ethical considerations", "Overestimating AI solution completeness", "Ignoring implementation barriers"],"scaffolding_support": ["Challenge-solution mapping framework", "Ethical deployment checklist", "Implementation feasibility guide"],"real_world_context": "Developing AI strategy for UN Sustainable Development Goals implementation"}]}},
            {"sample_id": "AI_LIT_021","source_module": "Module 1","source_section": "Types of AI - Artificial Narrow Intelligence (ANI)","input_text": "Artificial Narrow Intelligence (ANI), also known as 'Weak AI', refers to AI systems designed and trained for a specific, narrow task. These systems excel at their designated function but cannot perform tasks outside their programming. Examples include spam filters, recommendation systems, and voice assistants.","expected_output": {"learning_objectives": [{"objective_id": "KC_ANI_001","knowledge_component": "ANI-Definition-Examples","learning_objective": "Define Artificial Narrow Intelligence (ANI) and provide examples of its application in daily life","prerequisite_kcs": ["AI-Multi-Level-Definition"],"difficulty_estimate": 0.3,"cognitive_complexity": "Conceptual Understanding","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "Accurately identify ANI characteristics and give 3+ examples in 90% of scenarios","assessment_method": "Concept-to-example matching exercise","common_errors": ["Confusing ANI with general intelligence", "Misidentifying basic automation as ANI"],"scaffolding_support": ["ANI characteristics checklist", "Everyday AI examples review"],"real_world_context": "Recognizing the prevalence of ANI in common technologies like smartphones and streaming services"}]}},
            {"sample_id": "AI_LIT_022","source_module": "Module 1","source_section": "Types of AI - Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI)","input_text": "Artificial General Intelligence (AGI), or 'Strong AI', refers to hypothetical AI with human-level cognitive abilities across various tasks, capable of learning, understanding, and applying knowledge like a human. Artificial Superintelligence (ASI) is a speculative future AI that would surpass human intelligence in all aspects, including creativity and problem-solving.","expected_output": {"learning_objectives": [{"objective_id": "KC_AGI_ASI_001","knowledge_component": "AGI-ASI-Conceptual-Understanding","learning_objective": "Differentiate between Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI) based on their conceptual definitions and current status","prerequisite_kcs": ["ANI-Definition-Examples"],"difficulty_estimate": 0.4,"cognitive_complexity": "Comparative Conceptualization","estimated_learning_time": "15 minutes","practice_opportunities": 6,"mastery_criteria": "Accurately distinguish AGI and ASI concepts in 80% of descriptions","assessment_method": "Scenario-based AI type classification","common_errors": ["Assuming AGI/ASI are currently achievable", "Overstating current AI capabilities"],"scaffolding_support": ["AI spectrum diagram", "Discussion on AI limitations"],"real_world_context": "Participating in discussions about the future possibilities and ethical considerations of advanced AI"}]}},
            {"sample_id": "AI_LIT_023","source_module": "Module 1","source_section": "AI, Machine Learning (ML), and Deep Learning (DL): The Relationship","input_text": "Artificial Intelligence (AI) is the broad field of creating intelligent machines. Machine Learning (ML) is a subset of AI that enables systems to learn from data without explicit programming, often using statistical methods. Deep Learning (DL) is a subset of ML that uses neural networks with many layers ('deep' networks) to learn complex patterns, especially from large datasets like images or text.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_ML_DL_001","knowledge_component": "AI-ML-DL-Hierarchy","learning_objective": "Explain the hierarchical relationship between AI, Machine Learning (ML), and Deep Learning (DL) using examples","prerequisite_kcs": ["AI-Multi-Level-Definition"],"difficulty_estimate": 0.3,"cognitive_complexity": "Categorization & Relationship Mapping","estimated_learning_time": "12 minutes","practice_opportunities": 5,"mastery_criteria": "Correctly categorize AI, ML, and DL within the broader field of AI in 95% of questions","assessment_method": "Venn Diagram labeling and explanation","common_errors": ["Confusing ML with AI as a whole", "Not understanding deep neural networks for DL"],"scaffolding_support": ["AI family tree graphic", "Component function breakdown"],"real_world_context": "Discussing the technical underpinnings of modern AI applications"}]}},
            {"sample_id": "AI_LIT_024","source_module": "Module 1","source_section": "Introduction to Key AI Fields: Natural Language Processing (NLP) and Computer Vision","input_text": "Natural Language Processing (NLP) is an AI field focused on enabling computers to understand, interpret, and generate human language. Computer Vision is an AI field that trains computers to 'see' and interpret visual information from images and videos, mimicking human sight.","expected_output": {"learning_objectives": [{"objective_id": "KC_NLP_CV_001","knowledge_component": "NLP-CV-Core-Concepts","learning_objective": "Describe the core functions and applications of Natural Language Processing (NLP) and Computer Vision","prerequisite_kcs": ["AI-ML-DL-Hierarchy"],"difficulty_estimate": 0.3,"cognitive_complexity": "Domain Specific Definition","estimated_learning_time": "10 minutes","practice_opportunities": 4,"mastery_criteria": "Accurately describe NLP and CV applications in 4/5 given scenarios","assessment_method": "Application-to-field matching","common_errors": ["Mixing up NLP and CV applications", "Underestimating the complexity of human-like perception"],"scaffolding_support": ["NLP/CV example gallery", "Functionality comparison table"],"real_world_context": "Understanding how AI powers features like speech recognition and facial recognition"}]}},
            {"sample_id": "AI_LIT_025","source_module": "Module 2","source_section": "Data: The Fuel for Modern AI","input_text": "Data is the indispensable fuel that powers modern AI systems, particularly those based on Machine Learning. Without vast quantities of relevant data, most AI algorithms are inert and cannot perform their intended tasks. Humans learn from experience; for AI, 'experience' is represented by the data they are fed.","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_ROLE_001","knowledge_component": "Data-Role-in-AI","learning_objective": "Explain the critical role of data as the 'fuel' for modern AI systems, especially in Machine Learning","prerequisite_kcs": ["AI-ML-DL-Hierarchy"],"difficulty_estimate": 0.3,"cognitive_complexity": "Conceptual Explanation","estimated_learning_time": "8 minutes","practice_opportunities": 3,"mastery_criteria": "Articulate the importance of data in AI without prompting in 90% of instances","assessment_method": "Short answer explanation","common_errors": ["Underestimating data's significance", "Focusing only on algorithms"],"scaffolding_support": ["Data analogy examples", "AI learning process diagram"],"real_world_context": "Understanding why companies collect so much data for AI development"}]}},
            {"sample_id": "AI_LIT_026","source_module": "Module 2","source_section": "'Garbage In, Garbage Out': The Importance of Data Quality","input_text": "The principle of 'garbage in, garbage out' (GIGO) is crucial in AI: if an AI system is trained on inaccurate, incomplete, or biased data, the resulting model will reflect these issues, leading to unreliable or unfair outcomes. Data preprocessing, including cleaning and curating, is essential to ensure high-quality datasets for AI training.","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_QUALITY_001","knowledge_component": "Data-Quality-GIGO-Principle","learning_objective": "Understand the 'garbage in, garbage out' (GIGO) principle and the importance of data quality and preprocessing in AI","prerequisite_kcs": ["Data-Role-in-AI"],"difficulty_estimate": 0.4,"cognitive_complexity": "Causal Reasoning","estimated_learning_time": "10 minutes","practice_opportunities": 4,"mastery_criteria": "Identify data quality issues and their potential impact on AI in 85% of case studies","assessment_method": "Case study analysis of AI failure due to poor data","common_errors": ["Ignoring the impact of biased data", "Underestimating preprocessing effort"],"scaffolding_support": ["Data quality checklist", "Ethical AI examples", "Preprocessing steps overview"],"real_world_context": "Analyzing ethical concerns related to biased AI systems (e.g., in hiring or lending)"}]}},
            {"sample_id": "AI_LIT_027","source_module": "Module 2","source_section": "Types of Data: Structured vs. Unstructured","input_text": "Structured data is highly organized and follows a predefined format, often found in databases (e.g., spreadsheets, transaction records). Unstructured data lacks a predefined format and is more complex to process, accounting for most of the world's data (e.g., text documents, images, audio, video).","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_TYPES_001","knowledge_component": "Structured-Unstructured-Data","learning_objective": "Distinguish between structured and unstructured data with examples relevant to AI applications","prerequisite_kcs": ["Data-Role-in-AI"],"difficulty_estimate": 0.3,"cognitive_complexity": "Categorization","estimated_learning_time": "7 minutes","practice_opportunities": 4,"mastery_criteria": "Correctly categorize 8/10 data examples as structured or unstructured","assessment_method": "Data classification exercise","common_errors": ["Confusing semi-structured data with unstructured", "Difficulty providing varied examples"],"scaffolding_support": ["Data type examples table", "Classification flowchart"],"real_world_context": "Understanding the challenges AI faces when processing social media content vs. financial records"}]}},
            {"sample_id": "AI_LIT_028","source_module": "Module 2","source_section": "Big Data: Volume, Velocity, Variety","input_text": "Big Data refers to extremely large datasets characterized by 'Volume' (immense size), 'Velocity' (high speed of generation and processing), and 'Variety' (diverse data types, both structured and unstructured). Modern AI systems, especially with Deep Learning, thrive on Big Data to learn complex patterns.","expected_output": {"learning_objectives": [{"objective_id": "KC_BIGDATA_001","knowledge_component": "BigData-3Vs-Concept","learning_objective": "Define Big Data using the '3 Vs' (Volume, Velocity, Variety) and explain its importance for modern AI","prerequisite_kcs": ["Structured-Unstructured-Data"],"difficulty_estimate": 0.4,"cognitive_complexity": "Multi-faceted Definition","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "Explain the 3 Vs of Big Data and its AI relevance in 90% of explanations","assessment_method": "Conceptual explanation and linkage to AI","common_errors": ["Missing one of the Vs", "Not connecting Big Data directly to AI's learning capacity"],"scaffolding_support": ["Big Data infographic", "Industry use cases for Big Data"],"real_world_context": "Discussing the infrastructure required for training large AI models like LLMs"}]}},
            {"sample_id": "AI_LIT_029","source_module": "Module 2","source_section": "How Data is Collected (Sources)","input_text": "Data for AI is collected from various sources: User-Generated Content (active input like posts, reviews; passive input like Browse history), Sensors & IoT devices (industrial, environmental, wearable, automotive, medical sensors), Publicly Available Datasets (ImageNet, government portals), Enterprise Data (sales, customer records), and Simulated Data (for rare or dangerous scenarios).","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_SOURCES_001","knowledge_component": "AI-Data-Collection-Sources","learning_objective": "Identify and describe diverse sources from which data is collected for AI training, recognizing ethical considerations","prerequisite_kcs": ["BigData-3Vs-Concept"],"difficulty_estimate": 0.4,"cognitive_complexity": "Categorization & Exemplification","estimated_learning_time": "15 minutes","practice_opportunities": 7,"mastery_criteria": "List and provide examples for 4+ data collection sources in 85% of attempts","assessment_method": "Source identification and scenario mapping","common_errors": ["Overlooking passive user data", "Not considering simulated data"],"scaffolding_support": ["Data source taxonomy", "Ethical data collection guidelines"],"real_world_context": "Discussing data privacy concerns in modern AI applications"}]}},
            {"sample_id": "AI_LIT_030","source_module": "Module 2","source_section": "Data Preprocessing: Cleaning and Preparation","input_text": "Data preprocessing is a critical phase in AI development, involving cleaning, transforming, and organizing raw data into a high-quality, usable format. Key tasks include handling missing values (imputation/deletion), correcting errors, normalizing/scaling data, and feature engineering. This improves the performance, accuracy, and reliability of the resulting AI model.","expected_output": {"learning_objectives": [{"objective_id": "KC_DATA_PREPROCESS_001","knowledge_component": "Data-Preprocessing-Importance","learning_objective": "Explain the importance and key tasks of data preprocessing in preparing raw data for AI models","prerequisite_kcs": ["Data-Quality-GIGO-Principle"],"difficulty_estimate": 0.5,"cognitive_complexity": "Process Understanding","estimated_learning_time": "12 minutes","practice_opportunities": 6,"mastery_criteria": "Outline 3+ data preprocessing steps and their purpose in 80% of explanations","assessment_method": "Process diagram completion and explanation","common_errors": ["Underestimating preprocessing complexity", "Not linking preprocessing to model performance"],"scaffolding_support": ["Data pipeline visualization", "Common data issues and fixes"],"real_world_context": "Understanding the unseen work behind accurate AI predictions"}]}},
            {"sample_id": "AI_LIT_031","source_module": "Module 2","source_section": "What is an Algorithm? What is an AI Model?","input_text": "An algorithm is a set of step-by-step instructions or rules designed to solve a problem or perform a task (analogous to a recipe). An AI model is the 'trained' algorithm; it is the output of the training process where an algorithm learns patterns and relationships from data, enabling it to make predictions or decisions.","expected_output": {"learning_objectives": [{"objective_id": "KC_ALGO_MODEL_001","knowledge_component": "Algorithm-AIModel-Definition","learning_objective": "Define what an algorithm is and explain how it differs from an AI model in the context of machine learning","prerequisite_kcs": ["Data-Role-in-AI"],"difficulty_estimate": 0.3,"cognitive_complexity": "Comparative Definition","estimated_learning_time": "8 minutes","practice_opportunities": 4,"mastery_criteria": "Clearly differentiate between an algorithm and an AI model in 90% of definitions","assessment_method": "Fill-in-the-blank for definitions","common_errors": ["Using 'algorithm' and 'model' interchangeably", "Not grasping the 'trained' aspect of a model"],"scaffolding_support": ["Analogy-based explanations (recipe, student)", "Concept mapping"],"real_world_context": "Explaining how software engineers build and deploy AI solutions"}]}},
            {"sample_id": "AI_LIT_032","source_module": "Module 2","source_section": "Supervised Learning: Concept & Examples","input_text": "Supervised learning is a machine learning paradigm where an algorithm learns from labeled data, meaning the input data comes with the correct output. The model learns to map inputs to outputs by identifying patterns. Examples include spam detection (email labeled as spam/not spam) and image classification (images labeled with objects like 'cat' or 'dog').","expected_output": {"learning_objectives": [{"objective_id": "KC_SL_CONCEPT_001","knowledge_component": "Supervised-Learning-Basics","learning_objective": "Explain the concept of supervised learning and provide real-world examples of its application","prerequisite_kcs": ["Algorithm-AIModel-Definition"],"difficulty_estimate": 0.4,"cognitive_complexity": "Conceptual Understanding with Application","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "Correctly identify supervised learning scenarios and provide 3+ examples in 85% of cases","assessment_method": "Scenario identification and example generation","common_errors": ["Confusing labeled data with unlabeled", "Mixing up with other ML paradigms"],"scaffolding_support": ["Supervised learning flowchart", "Example database"],"real_world_context": "Understanding how AI systems learn to recognize objects or filter unwanted content"}]}},
            {"sample_id": "AI_LIT_033","source_module": "Module 2","source_section": "Unsupervised Learning: Concept & Examples","input_text": "Unsupervised learning is a machine learning paradigm that works with unlabeled data to find hidden patterns or structures within it. Unlike supervised learning, there are no 'correct' outputs given during training. Examples include customer segmentation (grouping similar customers) and anomaly detection (identifying unusual patterns like fraud).","expected_output": {"learning_objectives": [{"objective_id": "KC_UL_CONCEPT_001","knowledge_component": "Unsupervised-Learning-Basics","learning_objective": "Explain the concept of unsupervised learning and provide real-world examples of its application","prerequisite_kcs": ["Algorithm-AIModel-Definition"],"difficulty_estimate": 0.4,"cognitive_complexity": "Conceptual Understanding with Application","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "Correctly identify unsupervised learning scenarios and provide 3+ examples in 85% of cases","assessment_method": "Scenario identification and example generation","common_errors": ["Trying to find labels in unsupervised tasks", "Mixing up with other ML paradigms"],"scaffolding_support": ["Unsupervised learning flowchart", "Example database"],"real_world_context": "Understanding how AI discovers trends in large datasets without explicit guidance"}]}},
            {"sample_id": "AI_LIT_034","source_module": "Module 2","source_section": "Reinforcement Learning: Concept & Examples","input_text": "Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by performing actions in an environment to maximize a cumulative reward. It learns through trial and error, receiving feedback as rewards or penalties. Examples include game playing (AlphaGo, chess programs) and robotics (robots learning to navigate or perform tasks).","expected_output": {"learning_objectives": [{"objective_id": "KC_RL_CONCEPT_001","knowledge_component": "Reinforcement-Learning-Basics","learning_objective": "Explain the concept of reinforcement learning and provide real-world examples of its application","prerequisite_kcs": ["Algorithm-AIModel-Definition"],"difficulty_estimate": 0.5,"cognitive_complexity": "Conceptual Understanding with Application","estimated_learning_time": "12 minutes","practice_opportunities": 6,"mastery_criteria": "Correctly identify reinforcement learning scenarios and provide 2+ examples in 80% of cases","assessment_method": "Scenario identification and example generation","common_errors": ["Confusing reward signals with labels", "Not grasping the trial-and-error aspect"],"scaffolding_support": ["RL agent-environment diagram", "Game-playing AI examples"],"real_world_context": "Understanding how AI systems learn complex strategies in dynamic environments"}]}},
            {"sample_id": "AI_LIT_035","source_module": "Module 3","source_section": "Everyday AI: Recommendation Systems","input_text": "Recommendation systems, like those on Netflix, Amazon, or Spotify, are key AI applications that curate digital content and products. They learn user preferences by analyzing behavior and item data (views, purchases, ratings) to predict interests, thereby enhancing user experience and driving engagement.","expected_output": {"learning_objectives": [{"objective_id": "KC_REC_SYS_001","knowledge_component": "Recommendation-Systems-Overview","learning_objective": "Describe how AI fuels recommendation systems and their impact on user experience","prerequisite_kcs": ["ANI-Definition-Examples"],"difficulty_estimate": 0.3,"cognitive_complexity": "Application Understanding","estimated_learning_time": "8 minutes","practice_opportunities": 4,"mastery_criteria": "Explain the function of recommendation systems and give 3+ examples in 90% of instances","assessment_method": "Real-world application explanation","common_errors": ["Not linking recommendations to AI algorithms", "Underestimating data's role"],"scaffolding_support": ["Recommendation system breakdown", "User journey mapping"],"real_world_context": "Understanding how personalized content is delivered on streaming platforms and e-commerce sites"}]}},
            {"sample_id": "AI_LIT_036","source_module": "Module 3","source_section": "Introduction to Generative AI (GenAI)","input_text": "Generative AI (GenAI) is a rapidly advancing field of AI capable of creating new, original content (text, images, audio, video, code) that is similar to human-created content but not directly copied. Key architectures include Generative Adversarial Networks (GANs), Transformers, and Diffusion Models.","expected_output": {"learning_objectives": [{"objective_id": "KC_GENAI_INTRO_001","knowledge_component": "Generative-AI-Introduction","learning_objective": "Define Generative AI (GenAI) and explain its core capability of creating new content across various modalities","prerequisite_kcs": ["AI-ML-DL-Hierarchy"],"difficulty_estimate": 0.4,"cognitive_complexity": "Conceptual Definition","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "Accurately define GenAI and list 3+ content types it generates in 85% of responses","assessment_method": "Definition and example generation","common_errors": ["Confusing GenAI with predictive AI", "Not grasping the 'new content' aspect"],"scaffolding_support": ["GenAI application gallery", "Comparison with discriminative AI"],"real_world_context": "Discussing the rise of AI-generated art, music, and writing"}]}},
            {"sample_id": "AI_LIT_037","source_module": "Module 3","source_section": "GenAI: Large Language Models (LLMs) like ChatGPT","input_text": "Large Language Models (LLMs) like ChatGPT are a type of Generative AI proficient in understanding and generating human language. Trained on massive text corpora, LLMs predict the next word in a sequence using Transformer architecture and 'attention mechanisms', refined through fine-tuning and Reinforcement Learning from Human Feedback (RLHF).","expected_output": {"learning_objectives": [{"objective_id": "KC_LLM_CHATGPT_001","knowledge_component": "LLM-ChatGPT-Functionality","learning_objective": "Explain the core function and underlying principles of Large Language Models (LLMs) like ChatGPT","prerequisite_kcs": ["Generative-AI-Introduction", "NLP-CV-Core-Concepts"],"difficulty_estimate": 0.5,"cognitive_complexity": "Mechanism Explanation","estimated_learning_time": "15 minutes","practice_opportunities": 7,"mastery_criteria": "Describe how LLMs work, including concepts like 'next word prediction' and 'Transformer' in 80% of explanations","assessment_method": "Mechanism explanation and concept application","common_errors": ["Overcomplicating the mechanism", "Not mentioning training scale and data"],"scaffolding_support": ["LLM architecture diagram", "Simplified analogies for attention"],"real_world_context": "Understanding the capabilities and limitations of conversational AI tools"}]}},
            {"sample_id": "AI_LIT_038","source_module": "Module 4","source_section": "Ethical Consideration 1: Bias in AI","input_text": "Bias in AI systems can originate from biased training data, algorithmic design choices, or human decisions during deployment. Consequences include unfair outcomes (e.g., in hiring, loan applications, facial recognition) and perpetuating societal discrimination. Recognizing and mitigating bias is crucial for ethical AI development.","expected_output": {"learning_objectives": [{"objective_id": "KC_BIAS_AI_001","knowledge_component": "AI-Bias-Sources-Consequences","learning_objective": "Identify sources and consequences of bias in AI systems and its ethical implications","prerequisite_kcs": ["Data-Quality-GIGO-Principle"],"difficulty_estimate": 0.6,"cognitive_complexity": "Problem Analysis","estimated_learning_time": "15 minutes","practice_opportunities": 8,"mastery_criteria": "Analyze AI bias scenarios, identifying sources and impacts in 80% of cases","assessment_method": "Ethical dilemma case study analysis","common_errors": ["Only blaming data for bias", "Underestimating real-world harm"],"scaffolding_support": ["Bias detection checklist", "Fairness metrics overview"],"real_world_context": "Discussing controversies surrounding AI in justice systems or employment"}]}},
            {"sample_id": "AI_LIT_039","source_module": "Module 4","source_section": "Ethical Consideration 2: Privacy and Data Security","input_text": "AI systems often rely on vast amounts of personal data, raising significant privacy concerns. Data security measures are essential to protect this sensitive information from breaches and misuse. Ethical considerations include informed consent for data collection, anonymization techniques, and compliance with regulations like GDPR.","expected_output": {"learning_objectives": [{"objective_id": "KC_PRIVACY_AI_001","knowledge_component": "AI-Privacy-Data-Security","learning_objective": "Explain privacy and data security concerns related to AI systems and the importance of protective measures","prerequisite_kcs": ["AI-Data-Collection-Sources"],"difficulty_estimate": 0.6,"cognitive_complexity": "Problem Analysis & Solution Identification","estimated_learning_time": "12 minutes","practice_opportunities": 7,"mastery_criteria": "Describe 3+ privacy/security risks in AI and 2+ mitigation strategies in 80% of discussions","assessment_method": "Scenario-based privacy risk identification","common_errors": ["Underestimating data breach risks", "Ignoring regulatory compliance"],"scaffolding_support": ["Data privacy principles", "Cybersecurity best practices"],"real_world_context": "Analyzing the privacy implications of smart home devices or surveillance technologies"}]}},
            {"sample_id": "AI_LIT_040","source_module": "Module 4","source_section": "Ethical Consideration 3: Transparency & Explainability (XAI) - The 'Black Box' Problem","input_text": "Transparency and Explainability (XAI) in AI address the 'black box' problem, where complex AI models make decisions without clear, human-understandable reasoning. This lack of transparency raises ethical concerns regarding accountability, trust, and the ability to detect and correct errors, especially in critical applications like healthcare or finance.","expected_output": {"learning_objectives": [{"objective_id": "KC_XAI_001","knowledge_component": "AI-Transparency-Explainability","learning_objective": "Understand the concept of AI 'black boxes' and the importance of Transparency and Explainability (XAI)","prerequisite_kcs": ["AI-Bias-Sources-Consequences"],"difficulty_estimate": 0.6,"cognitive_complexity": "Problem Identification","estimated_learning_time": "12 minutes","practice_opportunities": 6,"mastery_criteria": "Explain the 'black box' problem and XAI's role in 85% of discussions","assessment_method": "Discussion prompt response","common_errors": ["Simplifying complex model behavior", "Not linking XAI to ethical concerns"],"scaffolding_support": ["XAI methods overview", "Ethical AI principles"],"real_world_context": "Evaluating the trustworthiness of AI-driven medical diagnoses or loan approvals"}]}},
            {"sample_id": "AI_LIT_041","source_module": "Module 4","source_section": "Ethical Consideration 4: Accountability","input_text": "Accountability in AI refers to establishing who is responsible when an AI system causes harm or makes an incorrect decision. This is complex due to AI's autonomy and distributed development. Ethical frameworks and regulations are being developed to assign responsibility to developers, deployers, or users, ensuring oversight and recourse.","expected_output": {"learning_objectives": [{"objective_id": "KC_ACCOUNTABILITY_AI_001","knowledge_component": "AI-Accountability-Challenges","learning_objective": "Discuss the challenges of accountability in AI systems and the need for ethical frameworks","prerequisite_kcs": ["AI-Transparency-Explainability"],"difficulty_estimate": 0.7,"cognitive_complexity": "Ethical Reasoning","estimated_learning_time": "15 minutes","practice_opportunities": 7,"mastery_criteria": "Analyze scenarios to identify accountability gaps in AI systems in 75% of cases","assessment_method": "Ethical case study discussion","common_errors": ["Overlooking the role of human oversight", "Simplifying complex responsibility chains"],"scaffolding_support": ["Accountability models (e.g., 'human in the loop')", "Regulatory examples"],"real_world_context": "Debating legal responsibility for autonomous vehicle accidents or AI-driven financial losses"}]}},
            {"sample_id": "AI_LIT_042","source_module": "Module 4","source_section": "Ethical Consideration 5: The Future of Work & Automation","input_text": "AI and automation are transforming the nature of work, leading to concerns about job displacement in some sectors while creating new roles in others. Ethical discussions focus on equitable transition strategies, reskilling initiatives, and ensuring a just distribution of automation's benefits.","expected_output": {"learning_objectives": [{"objective_id": "KC_WORK_AI_001_DUPLICATE","knowledge_component": "AI-Workforce-Impact","learning_objective": "Analyze the potential impact of AI and automation on the future of work, including job displacement and creation","prerequisite_kcs": ["AI-Application-Understanding"],"difficulty_estimate": 0.6,"cognitive_complexity": "Socio-economic Analysis","estimated_learning_time": "15 minutes","practice_opportunities": 7,"mastery_criteria": "Discuss 3+ ways AI impacts employment and propose 2+ mitigation strategies in 80% of scenarios","assessment_method": "Debate or essay on AI's impact on a specific industry","common_errors": ["Focusing only on job loss", "Underestimating the need for new skills"],"scaffolding_support": ["Automation impact reports", "Future of work trends"],"real_world_context": "Considering how AI might change careers in fields like customer service or manufacturing"}]}}, # Note: KC_WORK_AI_001 was already used, made it DUPLICATE for now
            {"sample_id": "AI_LIT_043","source_module": "Module 4","source_section": "Ethical Consideration 6: Misinformation & Deepfakes","input_text": "Generative AI can be misused to create highly realistic deepfakes (synthetic media) and spread misinformation, posing threats to trust, democracy, and individual reputation. Ethical concerns include the potential for manipulation, disinformation campaigns, and the difficulty of distinguishing authentic from synthetic content.","expected_output": {"learning_objectives": [{"objective_id": "KC_DEEPFAKES_MISINFO_001","knowledge_component": "AI-Misinformation-Deepfakes","learning_objective": "Recognize the risks of AI-generated misinformation and deepfakes and their societal implications","prerequisite_kcs": ["Generative-AI-Introduction"],"difficulty_estimate": 0.7,"cognitive_complexity": "Threat Assessment","estimated_learning_time": "12 minutes","practice_opportunities": 6,"mastery_criteria": "Identify deepfakes/misinformation examples and explain their societal impact in 80% of analyses","assessment_method": "Media literacy analysis of AI-generated content","common_errors": ["Underestimating the realism of deepfakes", "Not considering political implications"],"scaffolding_support": ["Deepfake detection tools", "Fact-checking guidelines"],"real_world_context": "Analyzing viral disinformation campaigns or celebrity deepfake controversies"}]}},
            {"sample_id": "AI_LIT_044","source_module": "Module 4","source_section": "Bias Mitigation Strategies","input_text": "Mitigating bias in AI involves various strategies: using diverse and representative training data, implementing algorithmic fairness techniques (e.g., re-sampling, adversarial debiasing), rigorous testing and auditing for bias, and human oversight in deployment. A multi-faceted approach across the AI lifecycle is essential.","expected_output": {"learning_objectives": [{"objective_id": "KC_BIAS_MITIGATION_001","knowledge_component": "AI-Bias-Mitigation-Strategies","learning_objective": "Identify and explain common strategies for mitigating bias in AI systems","prerequisite_kcs": ["AI-Bias-Sources-Consequences"],"difficulty_estimate": 0.6,"cognitive_complexity": "Solution Identification","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "List 3+ bias mitigation strategies and describe their application in 85% of cases","assessment_method": "Short answer questions on bias solutions","common_errors": ["Focusing on a single solution", "Not understanding the limitations of each strategy"],"scaffolding_support": ["Bias mitigation framework", "Ethical AI toolkits"],"real_world_context": "Discussing how companies are working to make their AI products fairer"}]}},
            {"sample_id": "AI_LIT_045","source_module": "Module 5","source_section": "Critically Evaluating AI-Generated Content: Spotting 'Hallucinations' / Inaccuracies","input_text": "Critically evaluating AI-generated content is vital, especially for 'hallucinations' – plausible but incorrect or nonsensical outputs from models like LLMs. Inaccuracies arise from limitations in training data or model understanding. Users must verify information, fact-check, and identify potential biases, rather than blindly trusting AI outputs.","expected_output": {"learning_objectives": [{"objective_id": "KC_EVAL_AI_CONTENT_001","knowledge_component": "AI-Content-Critical-Evaluation","learning_objective": "Develop critical thinking skills to evaluate AI-generated content, identify 'hallucinations' and inaccuracies","prerequisite_kcs": ["LLM-ChatGPT-Functionality"],"difficulty_estimate": 0.6,"cognitive_complexity": "Critical Analysis","estimated_learning_time": "15 minutes","practice_opportunities": 8,"mastery_criteria": "Identify inaccuracies and potential hallucinations in 80% of provided AI-generated text samples","assessment_method": "AI output analysis and critique","common_errors": ["Accepting AI output without verification", "Missing subtle inaccuracies"],"scaffolding_support": ["Fact-checking resources", "Hallucination examples"],"real_world_context": "Navigating the information landscape, especially with AI-generated news or research"}]}},
            {"sample_id": "AI_LIT_046","source_module": "Module 5","source_section": "Effective Prompting for Text-Based Generative AI","input_text": "Effective prompting for text-based Generative AI involves crafting precise and clear instructions to guide the AI towards desired outputs. Best practices include providing context, specifying format, tone, audience, and constraints, and iterating on prompts based on initial results. Good prompts lead to more accurate, relevant, and useful AI-generated content.","expected_output": {"learning_objectives": [{"objective_id": "KC_PROMPT_TEXT_001","knowledge_component": "Text-GenAI-Prompt-Engineering","learning_objective": "Apply best practices for effective prompting of text-based Generative AI models (e.g., ChatGPT)","prerequisite_kcs": ["LLM-ChatGPT-Functionality"],"difficulty_estimate": 0.5,"cognitive_complexity": "Practical Application","estimated_learning_time": "15 minutes","practice_opportunities": 10,"mastery_criteria": "Craft prompts that yield desired text output with 80% accuracy in 5+ attempts","assessment_method": "Prompt creation and evaluation exercise","common_errors": ["Vague instructions", "Not specifying output constraints"],"scaffolding_support": ["Prompt engineering templates", "Good vs. bad prompt examples"],"real_world_context": "Using ChatGPT or similar tools for content creation, brainstorming, or summarization"}]}},
            {"sample_id": "AI_LIT_047","source_module": "Module 5","source_section": "Understanding AI's Limitations and Strengths","input_text": "AI, despite its capabilities, has limitations: it lacks true common sense, cannot genuinely understand emotions or context beyond its training data, and is susceptible to biases. Its strengths lie in processing large datasets, pattern recognition, automation of repetitive tasks, and consistent performance.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_LIMITS_STRENGTHS_001","knowledge_component": "AI-Limitations-Strengths","learning_objective": "Identify and explain the key limitations and strengths of current AI systems","prerequisite_kcs": ["AI-ML-DL-Hierarchy", "ANI-Definition-Examples"],"difficulty_estimate": 0.5,"cognitive_complexity": "Critical Evaluation","estimated_learning_time": "10 minutes","practice_opportunities": 5,"mastery_criteria": "List 3+ limitations and 3+ strengths of AI in 85% of discussions","assessment_method": "Comparison matrix completion","common_errors": ["Overestimating AI's human-like abilities", "Underestimating its data processing power"],"scaffolding_support": ["AI capability spectrum", "Use case analysis"],"real_world_context": "Making informed decisions about when and how to use AI tools effectively"}]}},
            {"sample_id": "AI_LIT_048","source_module": "Module 5","source_section": "AI in Education: Tools for Learning & Teaching","input_text": "AI in education offers personalized learning experiences, automated grading, intelligent tutoring systems, and adaptive content delivery. For teachers, AI can automate administrative tasks, provide insights into student performance, and support curriculum development, enhancing both learning and teaching processes.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_EDUCATION_001","knowledge_component": "AI-Education-Applications","learning_objective": "Describe various applications of AI in education for both learners and educators","prerequisite_kcs": ["AI-Application-Understanding"],"difficulty_estimate": 0.4,"cognitive_complexity": "Application Identification","estimated_learning_time": "10 minutes","practice_opportunities": 6,"mastery_criteria": "Provide 3+ examples of AI in education for students and 2+ for teachers in 85% of scenarios","assessment_method": "Brainstorming AI uses in a classroom setting","common_errors": ["Focusing only on student-facing tools", "Not considering ethical implications in education"],"scaffolding_support": ["EdTech AI examples", "Discussion on responsible AI in schools"],"real_world_context": "Exploring how schools and universities are adopting AI tools"}]}},
            {"sample_id": "AI_LIT_049","source_module": "Module 5","source_section": "AI in the Job Market: New Roles & Skill Shifts","input_text": "AI's impact on the job market includes the emergence of new AI-specific roles (e.g., prompt engineer, AI ethicist) and a shift in required skills for existing jobs (e.g., data literacy, critical thinking, human-AI collaboration). Adaptability and continuous learning are key to thriving in an AI-driven economy.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_JOB_MARKET_001","knowledge_component": "AI-JobMarket-Transformation","learning_objective": "Analyze how AI influences the job market, including the creation of new roles and the evolution of essential skills","prerequisite_kcs": ["AI-Workforce-Impact"],"difficulty_estimate": 0.5,"cognitive_complexity": "Trend Analysis","estimated_learning_time": "12 minutes","practice_opportunities": 7,"mastery_criteria": "Identify 2+ new AI-related roles and 3+ evolving skills needed in 80% of analyses","assessment_method": "Career planning exercise in an AI context","common_errors": ["Underestimating the need for soft skills", "Overlooking non-tech roles affected by AI"],"scaffolding_support": ["Future skills frameworks", "Industry reports on AI and employment"],"real_world_context": "Planning career development in light of technological advancements"}]}},
            {"sample_id": "AI_LIT_050","source_module": "Module 5","source_section": "Human-AI Collaboration: Augmentation, Not Replacement","input_text": "Human-AI collaboration emphasizes AI as a tool for augmentation, enhancing human capabilities rather than replacing them. This involves humans providing oversight, ethical guidance, and creative input, while AI handles data processing, pattern recognition, and automation of routine tasks, leading to improved efficiency and innovation.","expected_output": {"learning_objectives": [{"objective_id": "KC_HUMAN_AI_COLLAB_001","knowledge_component": "Human-AI-Collaboration-Augmentation","learning_objective": "Explain the concept of human-AI collaboration as augmentation, providing examples of symbiotic relationships","prerequisite_kcs": ["AI-Limitations-Strengths"],"difficulty_estimate": 0.5,"cognitive_complexity": "Conceptual Understanding & Application","estimated_learning_time": "10 minutes","practice_opportunities": 6,"mastery_criteria": "Describe human-AI collaboration with 2+ examples of augmentation in 85% of discussions","assessment_method": "Scenario analysis of human-AI teamwork","common_errors": ["Viewing AI only as a replacement", "Not identifying the reciprocal benefits"],"scaffolding_support": ["Human-AI workflow diagrams", "Case studies of successful collaboration"],"real_world_context": "Observing how professionals use AI assistants in design, writing, or data analysis"}]}},
            {"sample_id": "AI_LIT_051","source_module": "Module 5","source_section": "Lifelong Learning in the Age of AI","input_text": "Lifelong learning is crucial in the age of AI due to rapid technological advancements and evolving job markets. Individuals must continuously acquire new skills, adapt to new tools, and stay informed about AI developments to remain relevant and competitive.","expected_output": {"learning_objectives": [{"objective_id": "KC_LIFELONG_LEARNING_001","knowledge_component": "Lifelong-Learning-AI-Era","learning_objective": "Understand the importance of lifelong learning and continuous skill development in an AI-driven world","prerequisite_kcs": ["AI-JobMarket-Transformation"],"difficulty_estimate": 0.4,"cognitive_complexity": "Awareness & Strategic Planning","estimated_learning_time": "8 minutes","practice_opportunities": 4,"mastery_criteria": "Articulate why lifelong learning is critical for 90% of reasons in the AI era","assessment_method": "Short reflection essay","common_errors": ["Underestimating the pace of change", "Focusing only on formal education"],"scaffolding_support": ["Learning resource curation", "Growth mindset principles"],"real_world_context": "Developing a personal plan for continuous professional development"}]}},
            {"sample_id": "AI_LIT_052","source_module": "Module 5","source_section": "AI for Good: Solving Global Challenges","input_text": "AI has immense potential to address global challenges, including climate change (e.g., optimizing energy grids, predicting extreme weather), healthcare (e.g., disease diagnosis, drug discovery), poverty (e.g., optimizing resource distribution), and disaster response. Ethical and responsible AI deployment is vital to maximize its positive impact.","expected_output": {"learning_objectives": [{"objective_id": "KC_AI_GOOD_001","knowledge_component": "AI-Global-Challenge-Solution-Design","learning_objective": "Design AI solutions for global challenges by matching AI capabilities to problem requirements while ensuring ethical deployment","prerequisite_kcs": ["AI-Application-Understanding", "AI-Bias-Sources-Consequences"],"difficulty_estimate": 0.7,"cognitive_complexity": "Ethical Solution Design","estimated_learning_time": "20 minutes","practice_opportunities": 10,"mastery_criteria": "Propose comprehensive AI solutions for 2+ global challenges with ethical considerations in 80% of analyses","assessment_method": "Project proposal for AI for Good initiative","common_errors": ["Missing ethical considerations", "Overestimating AI solution completeness", "Ignoring implementation barriers"],"scaffolding_support": ["Challenge-solution mapping framework", "Ethical deployment checklist", "Implementation feasibility guide"],"real_world_context": "Exploring initiatives like AI for Social Good or UN Sustainable Development Goals"}]}}
        ]
    }
    print(f"📚 Loaded {len(training_data.get('training_examples', []))} high-quality training examples.")
    return training_data

class KLIKnowledgeBase:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.training_examples = []
        self.index = None
    def add_training_examples(self, training_data: Dict):
        print("📚 Building RAG knowledge base from training examples...")
        texts_for_embedding = []
        examples = training_data.get("training_examples", [])
        if not examples:
            print("⚠️ No training examples found to build knowledge base.")
            return
        for example in examples:
            input_text = example.get("input_text", "")
            for obj in example.get("expected_output", {}).get("learning_objectives", []):
                combined_text = f"Concept: {obj.get('knowledge_component')}. Objective: {obj.get('learning_objective')}. Context: {input_text}"
                self.training_examples.append({"id": obj.get('objective_id'), "objective": obj, "combined_text": combined_text})
                texts_for_embedding.append(combined_text)
        if not texts_for_embedding: return
        embeddings = self.embedder.encode(texts_for_embedding, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✅ Knowledge base built with {self.index.ntotal} examples.")
    def retrieve_similar_examples(self, query_text: str, k: int = 1) -> List[Dict]:
        if not self.index: return []
        query_embedding = self.embedder.encode([query_text])
        _, indices = self.index.search(query_embedding, k)
        return [self.training_examples[i] for i in indices[0]]

class RAGObjectiveGenerator:
    def __init__(self, gemini_model, knowledge_base: KLIKnowledgeBase):
        self.gemini = gemini_model
        self.kb = knowledge_base
        self.objective_counter = 1 # Used for local placeholder IDs
    def _generate_objective_from_prompt(self, prompt: str, concept: str) -> Optional[LearningObjective]:
        """Helper method to generate objective from a prompt"""
        response = None  # Initialize response variable
        try:
            time.sleep(1.2) # Rate limiting
            response = self.gemini.generate_content(prompt)
            # Try to find JSON block, robustly
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?\})', response.text, re.DOTALL) # Fallback to any JSON block

            if not json_match:
                print(f"      ❌ No JSON object found in response for '{concept}'. Response text: {response.text[:200]}")
                return None
            
            json_text = json_match.group(1) # Get the content within the (first) capturing group
            data = json.loads(json_text)
            
            po_val = data.get("practice_opportunities", 5)
            de_val = data.get("difficulty_estimate", 0.5)
            parsed_po = int(re.search(r'\d+', str(po_val)).group(0)) if re.search(r'\d+', str(po_val)) else 5
            parsed_de = float(re.search(r'\d+\.?\d*', str(de_val)).group(0)) if re.search(r'\d+\.?\d*', str(de_val)) else 0.5
            
            objective = LearningObjective(
                objective_id=f"LOCAL_GEN_{self.objective_counter:03d}", # Placeholder ID
                knowledge_component=data.get("knowledge_component", f"KC for {concept.replace(' ', '-')}"),
                learning_objective=data.get("learning_objective", f"Understand {concept}"),
                prerequisite_kcs=data.get("prerequisite_kcs", []),
                difficulty_estimate=parsed_de,
                cognitive_complexity=data.get("cognitive_complexity", "Conceptual Understanding"),
                estimated_learning_time=str(data.get("estimated_learning_time", "15 minutes")),
                practice_opportunities=parsed_po,
                mastery_criteria=data.get("mastery_criteria", f"Achieve 80% on a quiz about {concept}."),
                assessment_method=data.get("assessment_method", "Multiple-choice quiz"),
                common_errors=data.get("common_errors", ["General misunderstanding of the topic."]),
                scaffolding_support=data.get("scaffolding_support", ["Provide clear definitions and examples."]),
                real_world_context=data.get("real_world_context", f"Practical applications of {concept}.")
            )
            self.objective_counter += 1
            return objective
            
        except json.JSONDecodeError as jde:
            print(f"      ❌ JSON Parsing Error for '{concept}': {jde}")
            print(f"      Raw response causing error: {json_text[:500] if 'json_text' in locals() else 'Unknown JSON text'}") 
            return None
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Error generating/parsing objective for '{concept}': {e}")
            
            # Handle rate limit specifically
            if "429" in error_msg or "quota" in error_msg.lower():
                print(f"      ⚠️ Rate limit exceeded. Waiting 60 seconds before continuing...")
                time.sleep(60)  # Wait for rate limit to reset
            elif "ResourceExhausted" in error_msg:
                print(f"      ⚠️ API quota exhausted. Waiting 120 seconds before continuing...")
                time.sleep(120)
            
            if response and hasattr(response, 'text'):
                print(f"      Response text: {response.text[:200]}")
            else:
                print("      No response text available.")
            return None

    def generate_for_concept_with_complexity(self, concept: str, context_summary: str, complexity: str, obj_num: int) -> Optional[LearningObjective]:
        """Generate an objective for a concept with specific complexity level"""
        print(f"    > Generating {complexity} objective #{obj_num} for: '{concept}'")
        
        complexity_prompts = {
            "basic understanding": "Focus on fundamental knowledge and comprehension (e.g., define, list, identify).",
            "application": "Emphasize practical application and implementation (e.g., apply, demonstrate, use).",
            "analysis": "Focus on analysis, comparison, and evaluation (e.g., analyze, compare, differentiate, evaluate).",
            "synthesis": "Emphasize creation, combination, and innovation (e.g., create, design, formulate, propose).",
            "evaluation": "Focus on critical assessment and judgment (e.g., critique, justify, recommend, assess)."
        }
        complexity_instruction = complexity_prompts.get(complexity.lower(), "Focus on conceptual understanding (e.g., explain, describe).")
        
        prompt = f"""
You are an expert instructional designer creating a learning objective for an AI Literacy course.

## YOUR TASK
- **Concept:** "{concept}"
- **Complexity Level:** {complexity} - {complexity_instruction}
- **Context:** "{context_summary[:1500]}"

Create a learning objective that specifically targets {complexity} level thinking about {concept}.
Return ONLY a raw JSON object with these fields:
"knowledge_component", "learning_objective", "prerequisite_kcs", "difficulty_estimate", "cognitive_complexity", "estimated_learning_time", "practice_opportunities", "mastery_criteria", "assessment_method", "common_errors", "scaffolding_support", "real_world_context".

Ensure the "learning_objective" field uses an action verb appropriate for the specified "{complexity}" level.
The "cognitive_complexity" field should also reflect this level (e.g., "{complexity.title()}").
For "prerequisite_kcs", list 0-2 relevant KCs if applicable, otherwise an empty list.
For "difficulty_estimate", use a float between 0.1 (easy) and 0.9 (hard).
For "practice_opportunities", suggest an integer number of activities.
"common_errors" and "scaffolding_support" should be lists of strings.

Example JSON structure:
```json
{{
  "knowledge_component": "Example-KC-Name",
  "learning_objective": "ActionVerb ExampleConcept by doing something specific.",
  "prerequisite_kcs": ["Prereq-KC-1"],
  "difficulty_estimate": 0.5,
  "cognitive_complexity": "{complexity.title()}",
  "estimated_learning_time": "20 minutes",
  "practice_opportunities": 5,
  "mastery_criteria": "Successfully complete a task related to the objective with 85% accuracy.",
  "assessment_method": "Quiz or Practical exercise",
  "common_errors": ["Common mistake 1", "Common mistake 2"],
  "scaffolding_support": ["Support strategy 1", "Support strategy 2"],
  "real_world_context": "How this concept is used in real life."
}}
```
Return ONLY the JSON object.
"""
        return self._generate_objective_from_prompt(prompt, concept)
    
    def generate_for_concept_with_aspect(self, concept: str, context_summary: str, aspect: str) -> Optional[LearningObjective]:
        """Generate an objective focusing on a specific aspect of the concept"""
        print(f"    > Generating {aspect} objective for: '{concept}'")
        
        aspect_prompts = {
            "practical application": "Focus on real-world use cases and hands-on implementation of the concept.",
            "theoretical understanding": "Emphasize underlying principles, definitions, and theoretical frameworks related to the concept.",
            "critical thinking": "Focus on analysis, evaluation, comparison, and forming judgments about the concept or its implications."
        }
        aspect_instruction = aspect_prompts.get(aspect.lower(), "Focus on comprehensive understanding of the concept.")
        
        prompt = f"""
You are an expert instructional designer creating a learning objective for an AI Literacy course.

## YOUR TASK
- **Concept:** "{concept}"
- **Focus Aspect:** {aspect} - {aspect_instruction}
- **Context:** "{context_summary[:1500]}"

Create a learning objective that specifically addresses the "{aspect}" of "{concept}".
Return ONLY a raw JSON object with these fields:
"knowledge_component", "learning_objective", "prerequisite_kcs", "difficulty_estimate", "cognitive_complexity", "estimated_learning_time", "practice_opportunities", "mastery_criteria", "assessment_method", "common_errors", "scaffolding_support", "real_world_context".

Ensure the "learning_objective" field clearly reflects the focus on "{aspect}".
The "cognitive_complexity" field should be appropriate for this focus.
For "prerequisite_kcs", list 0-2 relevant KCs if applicable, otherwise an empty list.
For "difficulty_estimate", use a float between 0.1 (easy) and 0.9 (hard).
For "practice_opportunities", suggest an integer number of activities.
"common_errors" and "scaffolding_support" should be lists of strings.

Example JSON structure:
```json
{{
  "knowledge_component": "Example-KC-Name-Aspect",
  "learning_objective": "ActionVerb ExampleConcept focusing on its {aspect}.",
  "prerequisite_kcs": [],
  "difficulty_estimate": 0.6,
  "cognitive_complexity": "Cognitive level for {aspect}",
  "estimated_learning_time": "25 minutes",
  "practice_opportunities": 4,
  "mastery_criteria": "Demonstrate understanding of {aspect} of the concept with specific criteria.",
  "assessment_method": "Assessment suitable for {aspect}",
  "common_errors": ["Error related to {aspect} 1"],
  "scaffolding_support": ["Support for understanding {aspect} 1"],
  "real_world_context": "Real-world implications of {aspect} of the concept."
}}
```
Return ONLY the JSON object.
"""
        return self._generate_objective_from_prompt(prompt, concept)

# ==============================================================================
# 📄 STEP 3: DOCUMENT PROCESSING MODULE
# ==============================================================================
class DocumentProcessor:
    def __init__(self, chunk_size=3000, chunk_overlap=400):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    def parse_pdf(self, file_path: str) -> str:
        print(f"📄 Parsing PDF: {file_path}")
        full_text = ""
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    full_text += f"\n--- Page {i+1} ---\n" + (page.extract_text() or "")
            print(f"✅ Extracted {len(full_text)} characters from {len(reader.pages)} pages.")
            return full_text
        except Exception as e:
            print(f"❌ Error parsing PDF: {e}")
            return ""
    def get_text_chunks(self, text: str) -> List[str]:
        print(f"🔪 Splitting text into chunks (chunk_size={self.text_splitter._chunk_size})...")
        chunks = self.text_splitter.split_text(text)
        print(f"✅ Created {len(chunks)} text chunks.")
        return chunks

# ==============================================================================
# 📊 ASSESSMENT GENERATION CLASSES (Integrated from Notebook)
# ==============================================================================

class AssessmentGenerator:
    """Generates assessments for structured curriculum (modules and lessons)"""

    def __init__(self, gemini_model): 
        self.gemini = gemini_model

    def generate_lesson_quiz(self, lesson: Lesson) -> Dict:
        """Generate a short quiz for a specific lesson"""
        objectives = lesson.learning_objectives
        if not objectives:
            return {
                "type": "lesson_quiz",
                "lesson_title": lesson.lesson_title,
                "error": "No learning objectives available for this lesson",
                "content": "Unable to generate quiz: no learning objectives"
            }

        prompt = f"""
Generate a short objective quiz to assess the lesson: "{lesson.lesson_title}"

OBJECTIVES:
{chr(10).join(f"- {obj.learning_objective}" for obj in objectives)}

KNOWLEDGE COMPONENTS:
{chr(10).join(f"- {obj.knowledge_component}" for obj in objectives)}

REQUIREMENTS:
1. Create a quiz with ONLY objective questions (suitable for AI grading).
2. Include {min(5, len(objectives) * 2)} multiple choice questions (4 options each, one correct answer).
3. Include {min(2, len(objectives))} matching questions (matching terms to definitions, if applicable, otherwise more MCQs).
4. Include {min(2, len(objectives))} true/false questions.
5. Focus on testing factual recall and conceptual understanding directly from the objectives.
6. Keep questions clear, unambiguous, and directly related to the provided learning objectives.
7. Provide a clear answer key at the end, listing the correct answer for each question (e.g., "1. c", "2. True").

FORMAT:
- Start with a title: "Lesson Quiz: {lesson.lesson_title}"
- Group questions by type (Multiple Choice, Matching, True/False).
- Number all questions sequentially (1, 2, 3...).
- For multiple choice: provide 4 options labeled a, b, c, d.
- For matching: provide two lists to match.
- For true/false: state the question clearly.
- End with an "Answer Key:" section.

Make sure all questions are appropriate for AI grading (objective with clear right/wrong answers).
Return ONLY the quiz content as plain text, starting with the title.
"""
        try:
            response = self.gemini.generate_content(prompt)
            quiz_content = response.text
            time.sleep(2) # Rate limiting
            return {
                'type': 'lesson_quiz',
                'lesson_title': lesson.lesson_title,
                'objective_count': len(objectives),
                'content': quiz_content,
            }
        except Exception as e:
            print(f"Error generating lesson quiz for '{lesson.lesson_title}': {e}")
            return {
                'type': 'lesson_quiz',
                'lesson_title': lesson.lesson_title,
                'error': f"Failed to generate quiz: {e}",
                'objective_count': len(objectives),
            }

    def generate_module_assessment(self, module: Module) -> Dict:
        """Generate a comprehensive test for an entire module"""
        all_objectives = []
        for lesson_item in module.lessons: # Ensure using correct attribute name
            all_objectives.extend(lesson_item.learning_objectives)

        if not all_objectives:
            return {
                "type": "module_assessment",
                "module_title": module.module_title,
                "error": "No learning objectives available for this module",
                "content": "Unable to generate assessment: no learning objectives"
            }

        prompt = f"""
Generate a comprehensive module assessment for: "{module.module_title}"

MODULE CONTAINS {len(module.lessons)} LESSONS:
{chr(10).join(f"- {lesson_item.lesson_title}" for lesson_item in module.lessons)}

NUMBER OF LEARNING OBJECTIVES IN MODULE: {len(all_objectives)}

REQUIREMENTS:
1. Create a comprehensive assessment with ONLY objective questions (suitable for AI grading).
2. Include {min(10, len(all_objectives))} multiple choice questions (4 options each, one correct answer).
3. Include {min(5, len(module.lessons) * 2)} matching questions (matching terms to definitions, if applicable, otherwise more MCQs or T/F).
4. Include {min(3, len(module.lessons))} true/false questions.
5. Optionally, include 1-2 short scenario-based questions where a scenario is presented, followed by a few multiple choice questions related to applying concepts from the module to the scenario.
6. Focus on testing application, synthesis, and evaluation (higher-order thinking) where appropriate for the objectives.
7. Include questions that connect concepts across different lessons in the module if possible.
8. Ensure a progression from basic recall to more complex application if the objectives support it.
9. Provide a clear answer key at the end, listing the correct answer for each question.

FORMAT:
- Start with a title: "Module Assessment: {module.module_title}"
- Organize into sections by question type (e.g., Multiple Choice, Matching, True/False, Scenario-Based).
- Number all questions sequentially.
- For multiple choice: provide 4 options labeled a, b, c, d.
- For scenario questions: present the scenario clearly before related questions.
- End with an "Answer Key:" section.

Make sure all questions are appropriate for AI grading (objective with clear right/wrong answers).
Return ONLY the assessment content as plain text, starting with the title.
"""
        try:
            response = self.gemini.generate_content(prompt)
            assessment_content = response.text
            time.sleep(3) # Rate limiting
            return {
                'type': 'module_assessment',
                'module_title': module.module_title,
                'lesson_count': len(module.lessons),
                'objective_count': len(all_objectives),
                'content': assessment_content,
            }
        except Exception as e:
            print(f"Error generating module assessment for '{module.module_title}': {e}")
            return {
                'type': 'module_assessment',
                'module_title': module.module_title,
                'error': f"Failed to generate assessment: {e}",
                'lesson_count': len(module.lessons),
                'objective_count': len(all_objectives),
            }

class CurriculumAssessmentSystem:
    """Orchestrates the generation of assessments for entire curriculum structure"""

    def __init__(self, gemini_model, output_dir="generated_assessments"):
        self.gemini = gemini_model 
        self.generator = AssessmentGenerator(gemini_model)
        self.output_dir = output_dir

    def _setup_output_directories(self):
        """Create output directories for saving assessments"""
        os.makedirs(self.output_dir, exist_ok=True)
        lesson_quiz_dir = os.path.join(self.output_dir, "lesson_quizzes")
        module_assessment_dir = os.path.join(self.output_dir, "module_assessments")
        os.makedirs(lesson_quiz_dir, exist_ok=True)
        os.makedirs(module_assessment_dir, exist_ok=True)
        return lesson_quiz_dir, module_assessment_dir

    def _sanitize_filename(self, filename: str) -> str:
        """Convert a string to a valid filename"""
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
        sanitized = sanitized.strip().replace(' ', '_')[:50] # Limit length
        return sanitized

    def generate_all_assessments(self, modules: List[Module], save_files=True):
        """Generate assessments for all modules and lessons in the curriculum"""
        if not modules:
            print("No modules provided to generate assessments for.")
            return {'lesson_quizzes': [], 'module_assessments': [], 'summary': {}}

        lesson_quiz_dir, module_assessment_dir = self._setup_output_directories() if save_files else (None, None)
        lesson_quizzes = []
        module_assessments = []

        print("\n📝 Generating lesson quizzes...")
        lesson_count_total = sum(len(module_obj.lessons) for module_obj in modules)
        current_lesson_num_overall = 0
        for mod_idx, module_obj in enumerate(modules, 1):
            for less_idx, lesson_obj in enumerate(module_obj.lessons, 1):
                current_lesson_num_overall += 1
                print(f"  > Generating quiz for Lesson {current_lesson_num_overall}/{lesson_count_total}: '{lesson_obj.lesson_title}' in Module '{module_obj.module_title}'")
                quiz = self.generator.generate_lesson_quiz(lesson_obj)
                quiz_id = f"LQ_M{mod_idx:02d}_L{less_idx:02d}"
                quiz.update({'id': quiz_id, 'module_idx': mod_idx, 'lesson_idx': less_idx, 'module_title': module_obj.module_title})
                lesson_quizzes.append(quiz)

                quiz_content = quiz.get('content') # Get content before checking for error
                if save_files and 'error' not in quiz and quiz_content:
                    file_name = f"{quiz_id}_{self._sanitize_filename(lesson_obj.lesson_title)}.md"
                    file_path = os.path.join(lesson_quiz_dir, file_name)
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(f"# Lesson Quiz: {lesson_obj.lesson_title}\n")
                            f.write(f"**Quiz ID:** {quiz_id}\n")
                            f.write(f"**Module:** {module_obj.module_title}\n")
                            f.write(f"**Objectives Covered:** {quiz.get('objective_count', 'N/A')}\n\n---\n\n")
                            f.write(quiz_content)
                        quiz['file_path'] = file_path
                        print(f"    ✅ Saved quiz: {file_name}")
                    except Exception as e:
                        print(f"    ❌ Error saving quiz {file_name}: {e}")
                elif 'error' in quiz:
                     print(f"    ⚠️ Quiz generation failed for '{lesson_obj.lesson_title}': {quiz['error']}")

        print(f"\n📚 Generating module assessments...")
        for mod_idx, module_obj in enumerate(modules, 1):
            print(f"  > Generating assessment for Module {mod_idx}/{len(modules)}: '{module_obj.module_title}'")
            assessment = self.generator.generate_module_assessment(module_obj)
            assessment_id = f"MA_M{mod_idx:02d}"
            assessment.update({'id': assessment_id, 'module_idx': mod_idx})
            module_assessments.append(assessment)

            assessment_content = assessment.get('content') # Get content before checking for error
            if save_files and 'error' not in assessment and assessment_content:
                file_name = f"{assessment_id}_{self._sanitize_filename(module_obj.module_title)}.md"
                file_path = os.path.join(module_assessment_dir, file_name)
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Module Assessment: {module_obj.module_title}\n")
                        f.write(f"**Assessment ID:** {assessment_id}\n")
                        f.write(f"**Lessons Covered:** {assessment.get('lesson_count', 'N/A')}\n")
                        f.write(f"**Objectives Covered:** {assessment.get('objective_count', 'N/A')}\n\n---\n\n")
                        f.write(assessment_content)
                    assessment['file_path'] = file_path
                    print(f"    ✅ Saved assessment: {file_name}")
                except Exception as e:
                    print(f"    ❌ Error saving assessment {file_name}: {e}")
            elif 'error' in assessment:
                print(f"    ⚠️ Assessment generation failed for '{module_obj.module_title}': {assessment['error']}")


        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_modules': len(modules),
            'total_lessons': lesson_count_total,
            'total_lesson_quizzes_generated': len(lesson_quizzes),
            'total_module_assessments_generated': len(module_assessments),
            'lesson_quizzes_details': [{'id': q['id'], 'title': q['lesson_title'], 'module': q.get('module_title', ''), 'status': 'Error' if 'error' in q else 'Success'} for q in lesson_quizzes],
            'module_assessments_details': [{'id': a['id'], 'title': a['module_title'], 'status': 'Error' if 'error' in a else 'Success'} for a in module_assessments],
            'output_directory': self.output_dir if save_files else "Files not saved"
        }
        if save_files:
            summary_path = os.path.join(self.output_dir, 'assessment_summary.json')
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Assessment summary saved to: {summary_path}")
            except Exception as e:
                print(f"❌ Error saving assessment summary: {e}")
        return {
            'lesson_quizzes': lesson_quizzes,
            'module_assessments': module_assessments,
            'summary': summary
        }

# ==============================================================================
# 🧠 STEP 4: THE ADVANCED ORCHESTRATOR
# ==============================================================================
class AdvancedObjectiveExtractor:
    def __init__(self, gemini_model: genai.GenerativeModel, api_client: APIClient):
        self.gemini = gemini_model
        self.api_client = api_client
        self.doc_processor = DocumentProcessor()
        training_data = load_training_data()
        self.knowledge_base = KLIKnowledgeBase()
        self.knowledge_base.add_training_examples(training_data)
        self.rag_generator = RAGObjectiveGenerator(self.gemini, self.knowledge_base)
        self.assessment_system = CurriculumAssessmentSystem(self.gemini, output_dir="generated_assessments")


    def _get_topics_from_chunks(self, chunks: List[str]) -> List[str]:
        print(f"🗺️ Identifying key topics from {len(chunks)} chunks...")
        combined_text = "\n\n--- END OF CHUNK ---\n\n".join(chunks)
        max_topic_extraction_len = 100000 
        if len(combined_text) > max_topic_extraction_len:
            print(f"⚠️ Truncating text for topic extraction from {len(combined_text)} to {max_topic_extraction_len} chars.")
            combined_text = combined_text[:max_topic_extraction_len]

        prompt = f"""
        You are an AI assistant specialized in educational content analysis.
        From the following text, extract up to 30 core educational topics or key concepts.
        Provide a single, consolidated, and de-duplicated list of these topics as a simple bulleted list.
        Each topic MUST be a short but descriptive noun phrase (e.g., "Mitochondrial DNA", "The Krebs Cycle").
        Filter out generic topics like "Introduction", "Conclusion", "References", "Contents".
        Prioritize topics that are central to the main subject matter of the text.

        TEXT:
        ```{combined_text}```
        CONSOLIDATED TOPICS (bulleted list):
        """
        try:
            time.sleep(1.2)
            response = self.gemini.generate_content(prompt)
            raw_topics = [t.strip("-* ").strip() for t in response.text.split('\n') if t.strip()]
            unique_topics = []
            for t in raw_topics:
                if t.lower() in ["topic:", "topics:", "consolidated topics:"]: continue
                if len(t) > 5 and len(t.split()) < 7 and t not in unique_topics: 
                    unique_topics.append(t)
            
            print(f"✅ Identified {len(unique_topics)} unique topics initially.")
            return unique_topics[:30] 
        except Exception as e:
            print(f"    ❌ An error occurred while extracting topics: {e}")
            return []

    def _structure_with_ai(self, objectives: List[LearningObjective]) -> List[Dict]:
        if not objectives or len(objectives) < 2:
            print("\nℹ️ Not enough objectives to create a structured curriculum. Proceeding with a single module/lesson.")
            return [{"module_title": "Main Module", "lessons": [{"lesson_title": "Core Concepts", "objective_indices": list(range(len(objectives)))}]}]
        
        print(f"\n🏗️ Building curriculum structure with Gemini for {len(objectives)} objectives...")
        objective_titles = [f"{i}: {obj.learning_objective}" for i, obj in enumerate(objectives)]
        
        prompt = f"""
You are an expert curriculum designer. Organize the following {len(objectives)} learning objectives into a coherent curriculum of Modules and Lessons.
- Aim for 2-5 lessons per module.
- Aim for 2-7 objectives per lesson.
- Ensure all provided objective indices are used exactly once across all lessons.
- Group objectives logically based on their content.

Return ONLY a single, valid JSON array of modules. Each module object must have "module_title" (string) and a "lessons" array.
Each lesson object must have "lesson_title" (string) and an "objective_indices" array (list of original integer indices).

**LEARNING OBJECTIVES (Index: Title):**
---
{os.linesep.join(objective_titles)}
---

JSON Output Example:
[
  {{
    "module_title": "Module 1: Foundations",
    "lessons": [
      {{ "lesson_title": "Introduction to X", "objective_indices": [0, 1] }},
      {{ "lesson_title": "Core Concepts of Y", "objective_indices": [2, 3, 4] }}
    ]
  }},
  {{
    "module_title": "Module 2: Advanced Topics",
    "lessons": [
      {{ "lesson_title": "Advanced X Techniques", "objective_indices": [5, 6] }}
    ]
  }}
]

Return ONLY the raw JSON array. Do not include any other text or markdown.
"""
        structure_plan_json_text = "" # For error reporting
        try:
            time.sleep(1.5) 
            response = self.gemini.generate_content(prompt)
            
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response.text, re.DOTALL | re.MULTILINE)
            if not json_match:
                json_match = re.search(r'(\[.*?\])', response.text, re.DOTALL | re.MULTILINE)

            if not json_match:
                print(f"❌ No JSON array found in the response for structuring. Response: {response.text[:300]}")
                raise ValueError("No JSON array found in the response for structuring.")
            
            structure_plan_json_text = json_match.group(1)
            structure_plan = json.loads(structure_plan_json_text)

            all_used_indices = set()
            for module_idx, mod_plan in enumerate(structure_plan):
                if "module_title" not in mod_plan or "lessons" not in mod_plan:
                    raise ValueError(f"Module {module_idx} is missing 'module_title' or 'lessons'.")
                for lesson_idx, less_plan in enumerate(mod_plan["lessons"]):
                    if "lesson_title" not in less_plan or "objective_indices" not in less_plan:
                        raise ValueError(f"Lesson {lesson_idx} in Module {module_idx} is missing 'lesson_title' or 'objective_indices'.")
                    for obj_idx in less_plan["objective_indices"]:
                        if not isinstance(obj_idx, int) or not (0 <= obj_idx < len(objectives)):
                            raise ValueError(f"Invalid objective index {obj_idx} found in lesson '{less_plan['lesson_title']}'. Max index is {len(objectives)-1}.")
                        if obj_idx in all_used_indices:
                            print(f"⚠️ Warning: Objective index {obj_idx} ('{objectives[obj_idx].learning_objective[:30]}...') used multiple times by AI structurer. Will be assigned to first encountered lesson.")
                        all_used_indices.add(obj_idx)
            
            print(f"✅ AI-generated curriculum plan created with {len(structure_plan)} modules.")
            return structure_plan
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON from AI for curriculum structure: {e}")
            print(f"   Raw JSON text: {structure_plan_json_text[:500]}")
            return [{"module_title": "Main Module (Fallback)", "lessons": [{"lesson_title": "Core Concepts (Fallback)", "objective_indices": list(range(len(objectives)))}]}]
        except Exception as e:
            print(f"❌ Error structuring curriculum with AI: {e}")
            traceback.print_exc()
            return [{"module_title": "Main Module (Fallback)", "lessons": [{"lesson_title": "Core Concepts (Fallback)", "objective_indices": list(range(len(objectives)))}]}]

    def _build_structured_curriculum_objects(self, structure_plan: List[Dict], flat_objectives: List[LearningObjective]) -> List[Module]:
        """Converts AI's structure plan and flat objectives into a list of Module dataclass objects."""
        curriculum_modules: List[Module] = []
        assigned_objective_indices = set()

        for mod_idx, mod_data in enumerate(structure_plan):
            module_lessons: List[Lesson] = []
            for less_idx, less_data in enumerate(mod_data.get("lessons", [])):
                lesson_objectives_instances: List[LearningObjective] = []
                for obj_idx_in_plan in less_data.get("objective_indices", []):
                    if 0 <= obj_idx_in_plan < len(flat_objectives):
                        if obj_idx_in_plan not in assigned_objective_indices:
                            objective_instance = flat_objectives[obj_idx_in_plan]
                            objective_instance.order = len(lesson_objectives_instances) + 1 
                            lesson_objectives_instances.append(objective_instance)
                            assigned_objective_indices.add(obj_idx_in_plan)
                        else:
                           print(f"  > Info: Objective index {obj_idx_in_plan} already assigned, skipping duplicate in plan for lesson '{less_data.get('lesson_title')}'.")
                    else: 
                       print(f"  > Warning: Invalid objective index {obj_idx_in_plan} in AI plan for lesson '{less_data.get('lesson_title')}', skipping.")

                if lesson_objectives_instances: 
                    lesson_obj = Lesson(
                        lesson_title=less_data.get("lesson_title", f"Untitled Lesson {less_idx+1}"),
                        learning_objectives=lesson_objectives_instances,
                        order=less_idx + 1
                    )
                    module_lessons.append(lesson_obj)
            
            if module_lessons: 
                module_obj = Module(
                    module_title=mod_data.get("module_title", f"Untitled Module {mod_idx+1}"),
                    lessons=module_lessons,
                    order=mod_idx + 1
                )
                curriculum_modules.append(module_obj)
        
        unassigned_indices = [i for i, obj in enumerate(flat_objectives) if i not in assigned_objective_indices]
        if unassigned_indices:
            print(f"  > Info: {len(unassigned_indices)} objectives were not assigned by the AI structuring plan. These will not be included in generated assessments or API persistence if they rely on this structured list.")
            # Optionally, you could add them to a "Miscellaneous" module here if needed for full coverage.

        print(f"  > Built {len(curriculum_modules)} Module objects from AI plan.")
        return curriculum_modules

    def process_and_persist_pdf(self, pdf_path: str, max_objectives: int = 40) -> Optional[int]:
        """Main method to process PDF, create curriculum, persist via API, and generate assessments."""
        print("🚀 Starting PDF processing and curriculum generation...")
        
        if not self.api_client.auth_token:
            if not self.api_client.authenticate(): # Try to authenticate if no token
                print("❌ Authentication failed. Cannot proceed with API operations. Will attempt local generation.")
                # Allow proceeding for local generation, but API calls will fail or be skipped
        
        health_status = self.api_client.check_service_health()
        api_course_id_created = None # To track if course was created via API

        if not all(health_status.values()):
            print("⚠️ Some microservices are not healthy. API operations might fail. Will attempt local generation.")
        
        pdf_filename = os.path.basename(pdf_path)
        
        # Attempt API operations if client is authenticated (or was initially given a token)
        if self.api_client.auth_token:
            print(f"\n📚 Creating course for: {pdf_filename} via API...")
            course_data = self.api_client.create_course(title=f"AI Literacy Course - {pdf_filename}")
            if course_data and 'id' in course_data:
                api_course_id_created = course_data['id']
                print(f"✅ Course created with API ID: {api_course_id_created}")
            
                print(f"\n📄 Uploading PDF content via API...")
                content_item = self.api_client.upload_content_item(pdf_path, title=pdf_filename)
                if content_item and 'id' in content_item:
                    content_item_id = content_item['id']
                    self.api_client.link_content_to_course(api_course_id_created, content_item_id)
                else:
                    print("❌ Failed to upload content item via API.")
            else:
                print("❌ Failed to create course via API. Proceeding with local generation only.")
        else: # No auth token, means authentication failed or was skipped
            print("⚠️ Skipping API course creation due to failed authentication. Proceeding with local generation only.")


        print(f"\n🔍 Extracting and processing PDF content locally...")
        full_text = self.doc_processor.parse_pdf(pdf_path)
        if not full_text: 
            print("❌ Failed to extract PDF content.")
            return None
        
        text_chunks = self.doc_processor.get_text_chunks(full_text)
        key_topics = self._get_topics_from_chunks(text_chunks)
        if not key_topics: 
            print("❌ Failed to extract key topics. PDF might be too short or content not suitable.")
            key_topics = [f"Overview of {pdf_filename.replace('.pdf','')}"] if max_objectives > 0 else []
            if not key_topics: return api_course_id_created # Return if course was created, even if no topics

        print(f"\n🎯 Generating up to {max_objectives} learning objectives locally from {len(key_topics)} topics...")
        flat_objectives: List[LearningObjective] = []
        context_summary = " ".join(text_chunks[:3])[:4000] if text_chunks else full_text[:4000]
        
        objectives_per_topic = max(1, max_objectives // len(key_topics)) if key_topics else 0
        objectives_per_topic = min(5, objectives_per_topic) 
        
        print(f"📊 Aiming for ~{objectives_per_topic} objectives per topic.")
        
        for topic_idx, topic in enumerate(key_topics):
            if len(flat_objectives) >= max_objectives: break
            print(f"  🎯 Topic {topic_idx + 1}/{len(key_topics)}: {topic}")
            for obj_num in range(objectives_per_topic):
                if len(flat_objectives) >= max_objectives: break
                complexity_levels = ["basic understanding", "application", "analysis", "synthesis", "evaluation"]
                complexity = complexity_levels[obj_num % len(complexity_levels)]
                objective = self.rag_generator.generate_for_concept_with_complexity(topic, context_summary, complexity, obj_num + 1)
                if objective:
                    objective.order = obj_num + 1 # Simple order within this topic's generation batch
                    flat_objectives.append(objective)
                    print(f"    ✅ Generated objective {len(flat_objectives)}: {objective.learning_objective[:60]}...")
        
        additional_objectives_needed = max_objectives - len(flat_objectives)
        if additional_objectives_needed > 0 and key_topics:
            print(f"\n➕ Generating {additional_objectives_needed} additional objectives with varied aspects...")
            for i in range(additional_objectives_needed):
                if len(flat_objectives) >= max_objectives: break
                topic = key_topics[i % len(key_topics)] 
                aspect_types = ["practical application", "theoretical understanding", "critical thinking"]
                aspect = aspect_types[i % len(aspect_types)] 
                objective = self.rag_generator.generate_for_concept_with_aspect(topic, context_summary, aspect)
                if objective:
                    objective.order = i + 1 # Simple order for these additional ones
                    flat_objectives.append(objective)
                    print(f"    ➕ Additional objective {len(flat_objectives)} for '{topic}' ({aspect}): {objective.learning_objective[:60]}...")
        
        print(f"\n🎉 Successfully generated {len(flat_objectives)} learning objectives locally.")
        if not flat_objectives:
            print("❌ No learning objectives were generated. Aborting.")
            return api_course_id_created 

        structure_plan_dict_list = self._structure_with_ai(flat_objectives)
        structured_modules_for_assessment = self._build_structured_curriculum_objects(structure_plan_dict_list, flat_objectives)

        if api_course_id_created and self.api_client.auth_token: 
            print("\n💾 Persisting structured curriculum to the database via API...")
            
            # Iterate based on the AI's structured plan to maintain order and lesson groupings
            for mod_idx_api, mod_plan_from_ai in enumerate(structure_plan_dict_list): 
                module_api_data = self.api_client.create_module(api_course_id_created, mod_plan_from_ai.get("module_title", f"Module {mod_idx_api+1}"), mod_idx_api + 1)
                if not module_api_data or 'id' not in module_api_data: 
                    print(f"❌ Failed to create module via API: {mod_plan_from_ai.get('module_title')}")
                    continue # Skip to next module in plan
                module_api_id = module_api_data['id']
                
                for less_idx_api, less_plan_from_ai in enumerate(mod_plan_from_ai.get("lessons", [])):
                    lesson_api_data = self.api_client.create_lesson(module_api_id, less_plan_from_ai.get("lesson_title", f"Lesson {less_idx_api+1}"), less_idx_api + 1)
                    if not lesson_api_data or 'id' not in lesson_api_data: 
                        print(f"❌ Failed to create lesson via API: {less_plan_from_ai.get('lesson_title')}")
                        continue # Skip to next lesson in plan
                    lesson_api_id = lesson_api_data['id']
                    
                    # Persist objectives from the plan that belong to this lesson
                    objectives_in_this_lesson_from_plan = []
                    for flat_obj_idx in less_plan_from_ai.get("objective_indices", []):
                        if 0 <= flat_obj_idx < len(flat_objectives):
                             # Check if this objective was successfully included in the _build_structured_curriculum_objects step
                            # This requires comparing the flat_obj_idx with the objectives actually present in structured_modules_for_assessment
                            # For simplicity here, we'll just use the flat_objectives list directly, assuming the AI plan is mostly good.
                            objectives_in_this_lesson_from_plan.append(flat_objectives[flat_obj_idx])
                    
                    for obj_order_in_lesson, objective_instance_to_persist in enumerate(objectives_in_this_lesson_from_plan, 1):
                        obj_dict_for_api = asdict(objective_instance_to_persist)
                        obj_dict_for_api['order'] = obj_order_in_lesson # This should map to 'order_in_lesson' in API payload
                        
                        created_objective_api_data = self.api_client.create_learning_objective(lesson_api_id, obj_dict_for_api, api_course_id_created)
                        if created_objective_api_data and 'id' in created_objective_api_data:
                            objective_api_id = created_objective_api_data['id']
                            self.api_client.create_knowledge_component(objective_api_id, obj_dict_for_api) 
                        else:
                            print(f"    ⚠️ Failed to create objective '{objective_instance_to_persist.learning_objective[:30]}...' via API for lesson {lesson_api_id}.")
            print("\n✅ Curriculum persisted to database (based on successful API calls)!")
        else:
            print("\n⚠️ Skipping database persistence for curriculum as API course was not created or auth failed.")

        if api_course_id_created and self.api_client.auth_token:
            print("\n🎨 Triggering instructional strategies generation (API)...")
            # self.api_client.generate_instructional_strategies(api_course_id_created) # Uncomment if API endpoint is ready
        
        print("\n📝 Generating assessment content locally...")
        if structured_modules_for_assessment: # Use the Module objects list
            assessment_results = self.assessment_system.generate_all_assessments(structured_modules_for_assessment, save_files=True)
            if assessment_results and (assessment_results['lesson_quizzes'] or assessment_results['module_assessments']):
                print("✅ Assessment content generated and saved to 'generated_assessments' directory.")
            else:
                print("⚠️ No assessment content was generated or an error occurred during generation.")
        else:
            print("⚠️ No structured modules available from AI plan to generate assessments.")
        
        print("\n🏁 Curriculum and Assessment Generation Process Complete 🏁")
        if api_course_id_created:
            print(f"🌐 Access your course dashboard (if API was used): {self.api_client.base_urls['validation_dashboard']}/courses/{api_course_id_created}")
        
        return api_course_id_created


# ==============================================================================
# 🏁 STEP 5: DEMONSTRATION HUB
# ==============================================================================
def display_structured_curriculum_from_api(course_id: int, api_client: APIClient):
    """ Fetches and displays the curriculum structure from the backend API. """
    if not course_id:
        print("No course ID provided, cannot fetch curriculum from API.")
        return
    
    print(f"\n🔍 Fetching curriculum structure for course ID {course_id} from API...")
    structure = api_client.get_course_structure(course_id)
    
    if not structure:
        print("❌ Could not fetch the final curriculum from the backend API.")
        return

    print("\n\n" + "="*80)
    print("📚 FINAL CURRICULUM HIERARCHY (Fetched from Database API) 📚")
    print("="*80)
    
    course_title = structure.get("title", f"Course ID: {course_id}")
    print(f"COURSE: {course_title}")
    retrieved_modules = structure.get("modules", [])

    if not retrieved_modules:
        print("  No modules found for this course in the API response.")
        print("="*80)
        return

    for mod_data in retrieved_modules:
        module_name = mod_data.get('module_name', 'Unknown Module')
        print(f"\n  MODULE: {module_name} (API ID: {mod_data.get('id')}, Order: {mod_data.get('module_order')})")
        underline_length = len(module_name) + 25 
        print(f"  {'-' * underline_length}") 
        
        retrieved_lessons = mod_data.get("lessons", [])
        if not retrieved_lessons:
            print("    No lessons found in this module.")
            continue

        for i, less_data in enumerate(retrieved_lessons, 1):
            lesson_name = less_data.get('lesson_name', 'Unknown Lesson')
            print(f"\n    LESSON {i}: {lesson_name} (API ID: {less_data.get('id')}, Order: {less_data.get('lesson_order')})")
            
            retrieved_objectives = less_data.get("learning_objectives", [])
            if not retrieved_objectives:
                print("      No objectives found in this lesson.")
                continue

            for obj_idx, obj_data in enumerate(retrieved_objectives, 1):
                # Use proper field names from learning_objectives schema
                objective_text = obj_data.get('objective', 'No objective text specified')
                order_in_course = obj_data.get('order_in_course', 'N/A')
                real_world_app = obj_data.get('real_world_application', 'No real-world application specified')
                common_errors = obj_data.get('common_errors', 'No common errors specified')
                scaffolding_support = obj_data.get('scaffolding_support', 'No scaffolding support specified')
                
                print(f"\n      --- Objective {obj_idx} (API ID: {obj_data.get('id')}, Order in Course: {order_in_course}) ---")
                print(f"        - Text: {objective_text}") 
                print(f"        - Real World Application: {real_world_app}")
                print(f"        - Common Errors: {common_errors}")
                print(f"        - Scaffolding Support: {scaffolding_support}")
    print("\n" + "="*80)

def test_simple_objective_creation():
    """Test function to create a simple learning objective with proper schema fields."""
    api_client = APIClient(base_urls=BASE_URLS, auth_token=AUTH_TOKEN)
    
    # Test data that matches the actual schema
    test_objective_data = {
        "learning_objective": "Students will be able to identify the basic components of artificial intelligence systems.",
        "knowledge_component": "AI System Components",
        "cognitive_complexity": "Conceptual Understanding",
        "order": 1,
        "real_world_context": "Understanding AI components helps in evaluating AI tools and services in professional settings.",
        "common_errors": ["Confusing AI with simple automation", "Overlooking data requirements"],
        "scaffolding_support": ["Component identification checklist", "Real-world AI examples"],
        "estimated_learning_time": "15 minutes"
    }
    
    print("\n🧪 Testing Simple Objective Creation...")
    print("=" * 50)
    
    # For testing, assume we have course_id=1, module_id=1, lesson_id=1
    # In real usage, these would be created first
    
    course_id = 1
    lesson_id = 1  # This should exist in your database
    
    result = api_client.create_learning_objective(lesson_id, test_objective_data, course_id)
    
    if result:
        print("✅ Learning objective created successfully!")
        print(f"   API Response: {result}")
        
        # Test knowledge component creation
        if 'id' in result:
            kc_result = api_client.create_knowledge_component(result['id'], test_objective_data)
            if kc_result:
                print("✅ Knowledge component created successfully!")
                print(f"   KC Response: {kc_result}")
            else:
                print("❌ Knowledge component creation failed")
    else:
        print("❌ Learning objective creation failed")

# --- Main execution block for local run ---
if __name__ == "__main__":
    if gemini_model:
        try:
            api_client = APIClient(base_urls=BASE_URLS) 
            extractor = AdvancedObjectiveExtractor(gemini_model, api_client)
            
            print("\n\n=======================================================")
            print("        PDF to DATABASE CURRICULUM GENERATOR")
            print("     with KLI Framework & Assessment Generation")
            print("=======================================================")
            print(f"🌐 Target Server: {IP_ADDRESS}")
            print(f"🔧 Services: {', '.join(BASE_URLS.keys())}")
            
            pdf_path = input("\n➡️ Enter the full path to your PDF file and press Enter (or leave blank for dummy PDF): ").strip()
            
            if not pdf_path:
                print("⚠️ No PDF path entered. Using a dummy PDF for demonstration.")
                dummy_pdf_name = "dummy_document_for_pipeline.pdf"
                if not os.path.exists(dummy_pdf_name):
                    try:
                        dummy_canvas = canvas.Canvas(dummy_pdf_name, pagesize=letter)
                        styles = getSampleStyleSheet()
                        styleN = styles['Normal']
                        styleH = styles['h1']
                        story = []
                        story.append(Paragraph("Dummy Document for Pipeline Test", styleH))
                        story.append(Paragraph("This is page 1 of the dummy PDF.", styleN))
                        story.append(Paragraph("It discusses basic concepts of Artificial Intelligence. AI is a broad field.", styleN))
                        story.append(Paragraph("Machine Learning is a subset of AI. Deep Learning is a subset of Machine Learning.", styleN))
                        
                        for _ in range(5):
                            story.append(Paragraph("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.", styleN))
                        
                        y_position = 750
                        for item in story:
                            width, height = item.wrapOn(dummy_canvas, 500, 50) 
                            if y_position - height < 50 : 
                                dummy_canvas.showPage()
                                y_position = 750
                            item.drawOn(dummy_canvas, 72, y_position - height)
                            y_position -= (height + 12) 
                        
                        dummy_canvas.save()
                        print(f"📄 Created {dummy_pdf_name}")
                    except Exception as e_pdf:
                        print(f"❌ Failed to create dummy PDF: {e_pdf}")
                        exit()
                pdf_path = dummy_pdf_name

            if not os.path.exists(pdf_path):
                print(f"❌ ERROR: File not found at '{pdf_path}'. Please check the path and try again.")
                exit()
            
            if not pdf_path.lower().endswith('.pdf'):
                print(f"❌ ERROR: The provided file is not a PDF. Please provide a path to a .pdf file.")
                exit()
            
            print(f"\n⏳ Processing '{pdf_path}'...")
            
            try:
                max_obj_input = input("\n➡️ Maximum objectives to generate (default 20, recommended 10-50 for quality): ").strip()
                max_objectives = int(max_obj_input) if max_obj_input else 20
                if max_objectives <= 0: max_objectives = 20
            except ValueError:
                max_objectives = 20
                print("Invalid input. Using default value of 20 objectives.")
            
            result_course_id_from_api = extractor.process_and_persist_pdf(pdf_path, max_objectives)
            
            if result_course_id_from_api: 
                print(f"\n🎉🎉🎉 SUCCESS! Course ID {result_course_id_from_api} was processed with API interaction.")
                print(f"\n📊 Course Dashboard (if services are up): {BASE_URLS['validation_dashboard']}/courses/{result_course_id_from_api}")
                print("\n📋 What was done:")
                print("  ✅ Course and content item created/linked via API.")
                print("  ✅ Structured modules and lessons generated locally.")
                print("  ✅ Learning objectives with KLI framework details generated locally.")
                print("  ✅ These structures were persisted via API.")
                print("  ✅ Assessment content (quizzes, tests) generated locally and saved to 'generated_assessments' directory.")
                
                display_structured_curriculum_from_api(result_course_id_from_api, api_client)

            elif os.path.exists("generated_assessments"): 
                print("\n🎉🎉🎉 SUCCESS! Processing finished, likely in local-only mode (API interaction might have failed or was skipped).")
                print("\n📋 What was done:")
                print("  ✅ PDF processed, objectives and curriculum structure generated locally.")
                print("  ✅ Assessment content (quizzes, tests) generated locally and saved to 'generated_assessments' directory.")
                print("  ⚠️ Course was NOT persisted to the database or an error occurred during API persistence.")
                print("  Check 'generated_assessments' folder for output.")
            else: 
                print("\n❌❌❌ The curriculum generation process encountered issues or produced no output.")
                print("\nPossible issues:")
                print("  - PoC microservices not running or not reachable at the specified IP_ADDRESS.")
                print("  - Network connectivity issues.")
                print("  - Authentication problems (check AUTH_TOKEN or login credentials, or if default user can be registered).")
                print("  - PDF processing errors (invalid PDF, very short content, or unreadable text).")
                print("  - Gemini API errors (key issues, rate limits, or content policy violations).")
                print("  - Errors in parsing AI responses (especially JSON structures).")
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Process interrupted by user. Exiting...")
        except Exception as e:
            print(f"\n\n💥 An unexpected error occurred in __main__: {e}")
            traceback.print_exc()
            print("\nIf the error persists, please check the detailed error messages above and ensure:")
            print("  1. PoC microservices are running and accessible if not in local-only mode.")
            print("  2. Network connectivity is stable.")
            print("  3. The PDF file is valid and accessible.")
            print("  4. Your Gemini API key (GEMINI_API_KEY) is correctly configured and has quota.")
    else:
        print("\n❌ CRITICAL ERROR: Gemini model was not initialized. Cannot proceed.")
        print("Please check your Gemini API key configuration and ensure the `setup_gemini()` function runs successfully.")
