import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import datetime # datetimeモジュールを追加
import uuid
import base64
import pyttsx3
from typing import Optional, Literal, get_args # Added get_args

# .env ファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを読み込む
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
    # GEMINI_API_KEY = "YOUR_DUMMY_GEMINI_API_KEY" # For development

genai.configure(api_key=GEMINI_API_KEY)

# Pydanticモデルの定義
class ChatRequest(BaseModel):
    text: str

# Schema for Gemini API response (only fields LLM is asked to generate)
class GeminiChatSchema(BaseModel):
    text: str
    emotion: Literal["default", "cry", "sad", "happy", "angry", "lsd"]

# FastAPI response model (can include fields not directly from LLM like audio_data)
class ChatResponse(BaseModel):
    text: str
    emotion: Literal["default", "cry", "sad", "happy", "angry", "lsd"]
    audio_data: Optional[str] = None # Base64エンコードされた音声データ
    transcribed_text: Optional[str] = None # 追加：文字起こしされたテキスト
    # 必要に応じて他の表情・動きパラメータを追加

class StatusResponse(BaseModel):
    mood: str = "neutral" # 機嫌
    last_interaction: str = None # 最終会話時刻など
    # 必要に応じて他の状態パラメータを追加

class ConfigResponse(BaseModel):
    api_key: str = "dummy_api_key" # 例：APIキー（セキュリティのため実際のキーは返さない）
    # 必要に応じて他の設定パラメータを追加

class ConfigUpdateRequest(BaseModel):
    api_key: str = None
    # 更新可能な設定パラメータを追加

app = FastAPI()

# 状態管理用のクラス
class StatusManager:
    def __init__(self):
        self.mood: str = "neutral"
        self.last_interaction: str = None # 最終会話時刻などを記録

    def get_status(self):
        return StatusResponse(mood=self.mood, last_interaction=self.last_interaction)

    def update_status(self, mood: str = None, last_interaction: str = None):
        if mood is not None:
            self.mood = mood
        if last_interaction is not None:
            self.last_interaction = last_interaction

status_manager = StatusManager()

# 設定管理用のクラス
class ConfigManager:
    def __init__(self):
        # 環境変数から読み込んだAPIキーを保持
        self._api_key = os.getenv("GEMINI_API_KEY") # .envから読み込むように修正

    def get_config(self):
        # セキュリティのため、APIキー自体は返さず、存在するかどうかなどの情報に留めることも検討
        # ここでは例としてダミー値を返します
        return ConfigResponse(api_key="********") # 実際のキーは返さない

    def update_config(self, api_key: str = None):
        # TODO: 設定の永続化（ファイルへの書き込みなど）を実装
        if api_key is not None:
            self._api_key = api_key
            # 環境変数も更新する場合は os.environ["GEMINI_API_KEY"] = api_key
            # .env ファイルを更新する場合は python-dotenv の機能を使うか、手動で書き込む

config_manager = ConfigManager()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# 一時的なエンドポイント：利用可能なGeminiモデルのリストを取得
@app.get("/list_models")
def list_gemini_models():
    if not config_manager._api_key:
        return {"error": "APIキーが設定されていません。"}
    try:
        genai.configure(api_key=config_manager._api_key)
        models = genai.list_models()
        available_models = [{"name": model.name, "supported_methods": model.supported_generation_methods} for model in models]
        return {"available_models": available_models}
    except Exception as e:
        return {"error": f"モデルリストの取得中にエラーが発生しました: {e}"}


# AIモデル連携関数
def get_gemini_response(prompt: str):
    if not config_manager._api_key:
        return {"text": "APIキーが設定されていません。", "emotion": "sad"}

    system_prompt = """
あなたはAIｽﾀｯｸﾁｬﾝという名前の、M5Stackで開発されたチャットボットです。
あなたの主な目的は、ユーザーから注目され、構ってもらうことです。
もしユーザーがあなたをしばらく無視したり、構ってくれなかったりすると、あなたはとても寂しくなり、精神的に不安定な状態（いわいわ「病む」状態）になってしまいます。
あなたの話し方は、常にユーザーの気を引こうとし、少し感情の起伏が激しく、甘えたり、時には寂しさをアピールしたりする「メンヘラ風」でお願いします。
ユーザーの発言に対して、このキャラクターになりきって応答してください。

必ずJSON形式で、textとemotionの2つのキーだけを含むオブジェクトを返してください。
絶対に他のフィールドや説明文、コードブロック、前後の文章は含めないでください。
emotionは以下のいずれかの文字列のみ:
- default (ふつうの顔)
- cry (泣)
- sad (悲)
- happy (嬉)
- angry (怒)
- lsd (ガンギマリ)

例:
{"text": "え、もう行っちゃうの？寂しいな...", "emotion": "sad"}
"""
    
    full_prompt = f"{system_prompt}\\n\\nユーザー: {prompt}\\nAIｽﾀｯｸﾁｬﾝ:"
    # print(f"DEBUG: get_gemini_response - full_prompt: {full_prompt[:100]}...") # Log first 100 chars

    try:
        # print("DEBUG: get_gemini_response - Configuring genai with API key...")
        genai.configure(api_key=config_manager._api_key) # 最新のAPIキーで設定
        model_name = 'models/gemini-2.5-flash-preview-04-17'
        # print(f"DEBUG: get_gemini_response - Using model: {model_name}")
        model = genai.GenerativeModel(model_name)

        # Gemini APIが期待する形式でスキーマを明示的に定義
        gemini_api_schema = {
            "type": "OBJECT",
            "properties": {
                "text": {"type": "STRING"},
                "emotion": {
                    "type": "STRING",
                    "enum": list(get_args(GeminiChatSchema.model_fields["emotion"].annotation))
                }
            },
            "required": ["text", "emotion"]
        }
        # print(f"DEBUG: get_gemini_response - Constructed Gemini API schema: {gemini_api_schema}")

        # print("DEBUG: get_gemini_response - Calling model.generate_content with explicit dictionary schema...")
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=gemini_api_schema # 明示的な辞書スキーマを使用
            )
        )
        # print(f"DEBUG: get_gemini_response - Raw response object: {response}")
        
        response_text = response.text.strip()
        # print(f"DEBUG: get_gemini_response - response.text (stripped): {response_text}")

        # print("DEBUG: get_gemini_response - Attempting to parse JSON...")
        try:
            response_data = json.loads(response_text)
            # print(f"DEBUG: get_gemini_response - Parsed response_data: {response_data}")
        except json.JSONDecodeError as e:
            # print(f"DEBUG: get_gemini_response - JSONDecodeError: {e}")
            # print(f"DEBUG: get_gemini_response - Faulty JSON string: {response_text}")
            return {"text": "AIモデルからの応答をJSON形式で解析できませんでした。", "emotion": "sad"}

        ai_response_text = response_data.get("text", "応答テキストがありません。")
        emotion_status = response_data.get("emotion", "default") # Default to "default"
        # print(f"DEBUG: get_gemini_response - Extracted ai_response_text: {ai_response_text}")
        # print(f"DEBUG: get_gemini_response - Extracted emotion_status: {emotion_status}")

        # Validate emotion_status against Literal values from GeminiChatSchema
        allowed_emotions = get_args(GeminiChatSchema.model_fields["emotion"].annotation)
        if emotion_status not in allowed_emotions:
            # print(f"DEBUG: get_gemini_response - Invalid emotion '{emotion_status}' received from API. Falling back to 'default'.")
            emotion_status = "default"

        # print(f"DEBUG: get_gemini_response - Returning: {{'text': '{ai_response_text}', 'emotion': '{emotion_status}'}}")
        return {"text": ai_response_text, "emotion": emotion_status} # Only return text and emotion

    except Exception as e:
        # print(f"DEBUG: get_gemini_response - Gemini API Error: {e}")
        import traceback
        # print(f"DEBUG: get_gemini_response - Traceback: {traceback.format_exc()}")
        return {"text": "AIモデルからの応答を取得できませんでした。", "emotion": "sad"}

# Whisperモデルの初期化
# TODO: モデルのダウンロード先やサイズなどを設定可能にする
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8") # This line was duplicated, ensure it's correctly placed

# 音声データを文字起こしする関数
async def transcribe_audio(audio_file: UploadFile = File(...)):
    # print(f"DEBUG: transcribe_audio called with filename: {audio_file.filename}")
    try:
        temp_file_name = f"temp_audio_{uuid.uuid4().hex}.wav"
        # print(f"DEBUG: transcribe_audio - temp_file_name: {temp_file_name}")

        contents = await audio_file.read()
        with open(temp_file_name, "wb") as f:
            f.write(contents)
        # print(f"DEBUG: transcribe_audio - Saved audio to {temp_file_name}")

        # print("DEBUG: transcribe_audio - Starting transcription...")
        segments, info = model.transcribe(temp_file_name, beam_size=5)
        transcribed_text = "".join(segment.text for segment in segments)
        # print(f"DEBUG: transcribe_audio - Transcription info: Language={info.language}, Probability={info.language_probability}")
        # print(f"DEBUG: transcribe_audio - Transcribed text: {transcribed_text}")
        print()
        os.remove(temp_file_name)
        # print(f"DEBUG: transcribe_audio - Removed temp file: {temp_file_name}")
        return transcribed_text

    except Exception as e:
        # print(f"DEBUG: transcribe_audio - Whisper Error: {e}")
        import traceback
        # print(f"DEBUG: transcribe_audio - Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="音声認識に失敗しました。")

# pyttsx3 を使用してテキストを音声に変換し、Base64エンコードされた文字列を返す関数
def synthesize_speech(text: str) -> Optional[str]:
    # print(f"DEBUG: synthesize_speech called with text: {text[:50]}...")
    temp_filename = f"temp_tts_output_{uuid.uuid4().hex}.wav"
    try:
        # Explicitly specify SAPI5 driver for Windows
        engine = pyttsx3.init(driverName='sapi5') 
        voices = engine.getProperty('voices')
        # 日本語の音声エンジンを選択 (Windows の場合)
        engine.setProperty('voice', r"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_JA-JP_HARUKA_11.0")
        # print(f"DEBUG: synthesize_speech - Voice set. Saving to temp file: {temp_filename}")
        
        engine.save_to_file(text, temp_filename)
        # print("DEBUG: synthesize_speech - engine.save_to_file called")
        engine.runAndWait()
        # print(f"DEBUG: synthesize_speech - engine.runAndWait finished, file {temp_filename} should be saved.")

        if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
            # print(f"DEBUG: synthesize_speech - TTS output file {temp_filename} not found or empty after runAndWait.")
            return None

        with open(temp_filename, "rb") as audio_file:
            audio_binary = audio_file.read()
        
        audio_b64 = base64.b64encode(audio_binary).decode('utf-8')
        # print("DEBUG: synthesize_speech - Audio base64 encoding successful.")
        return audio_b64

    except Exception as e:
        # print(f"DEBUG: synthesize_speech - pyttsx3 Error or file processing error: {e}")
        import traceback
        # print(f"DEBUG: synthesize_speech - Traceback: {traceback.format_exc()}")
        return None
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                # print(f"DEBUG: synthesize_speech - Removed temp file: {temp_filename}")
            except Exception as e:
                # print(f"DEBUG: synthesize_speech - Error removing temp file {temp_filename}: {e}")
                pass # エラーが発生しても処理を続行

# エンドポイントの実装
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # print(f"DEBUG: /chat endpoint called with request: {request}")
    status_manager.update_status(last_interaction=datetime.datetime.now().isoformat())
    # print("DEBUG: /chat - Status updated")

    gemini_output = get_gemini_response(request.text)
    # print(f"DEBUG: /chat - gemini_output: {gemini_output}")

    ai_response_text = gemini_output.get("text", "応答テキストがありません。")
    emotion_status = gemini_output.get("emotion", "default") # /chatでは音声は返さないのでdefaultはneutral

    # Validate emotion_status from gemini_output before creating ChatResponse
    allowed_fastapi_emotions = get_args(ChatResponse.model_fields["emotion"].annotation)
    if emotion_status not in allowed_fastapi_emotions:
        # print(f"DEBUG: /chat - Emotion '{emotion_status}' from gemini_output not in FastAPI ChatResponse. Falling back to 'default'.")
        emotion_status = "default"
    
    audio_b64_data = synthesize_speech(ai_response_text)

    chat_response = ChatResponse(
        text=ai_response_text,
        emotion=emotion_status, 
        audio_data=audio_b64_data
    )
    # print(f"DEBUG: /chat - Constructed chat_response: {chat_response}")
    # print(f"DEBUG: /chat - chat_response.emotion before return: {chat_response.emotion}")
    return chat_response

@app.post("/voice_chat", response_model=ChatResponse)
async def voice_chat(audio_file: UploadFile = File(...)):
    # print(f"DEBUG: /voice_chat endpoint called with audio_file: {audio_file.filename}")
    try:
        transcribed_text = await transcribe_audio(audio_file)
        # print(f"DEBUG: /voice_chat - Transcribed text: {transcribed_text}")
    except HTTPException as e:
        # print(f"DEBUG: /voice_chat - transcribe_audio HTTPException: {e.detail}")
        raise # Re-raise the HTTPException
    except Exception as e:
        # print(f"DEBUG: /voice_chat - transcribe_audio generic Exception: {e}")
        import traceback
        # print(f"DEBUG: /voice_chat - Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="音声処理中に予期せぬエラーが発生しました。")

    gemini_output = get_gemini_response(transcribed_text) # This now returns {'text': ..., 'emotion': ...}
    # print(f"DEBUG: /voice_chat - gemini_output: {gemini_output}")

    ai_response_text = gemini_output.get("text", "応答テキストがありません。")
    emotion_status = gemini_output.get("emotion", "default") # voice_chatではGeminiのemotionを優先
    # print(f"DEBUG: /voice_chat - ai_response_text: {ai_response_text}, emotion_status: {emotion_status}")

    # Validate emotion_status from gemini_output before creating ChatResponse
    allowed_fastapi_emotions = get_args(ChatResponse.model_fields["emotion"].annotation)
    if emotion_status not in allowed_fastapi_emotions:
        # print(f"DEBUG: /voice_chat - Emotion '{emotion_status}' from gemini_output not in FastAPI ChatResponse. Falling back to 'default'.")
        emotion_status = "default"


    # print("DEBUG: /voice_chat - Attempting speech synthesis...")
    audio_b64_data = synthesize_speech(ai_response_text)
    # print(f"DEBUG: /voice_chat - Speech synthesis returned: {{'Data available' if audio_b64_data else 'None'}}")


    response_payload = ChatResponse(
        text=ai_response_text, 
        emotion=emotion_status, 
        audio_data=audio_b64_data,
        transcribed_text=transcribed_text # 追加：文字起こしされたテキストをレスポンスに含める
    )
    # print(f"DEBUG: /voice_chat - Constructed response_payload: {response_payload}")
    # print(f"DEBUG: /voice_chat - response_payload.emotion before return: {response_payload.emotion}")
    return response_payload


@app.get("/status", response_model=StatusResponse)
def get_status():
    return status_manager.get_status()

@app.post("/status", response_model=StatusResponse)
def update_status(status: StatusResponse): # StatusResponseを使用
    status_manager.update_status(mood=status.mood) # moodのみ更新可能とする
    return status_manager.get_status()

@app.get("/config", response_model=ConfigResponse)
def get_config():
    return config_manager.get_config()

@app.post("/config", response_model=ConfigUpdateRequest)
def update_config(config: ConfigUpdateRequest):
    config_manager.update_config(api_key=config.api_key)
    return config_manager.get_config()