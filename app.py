import traceback
import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import datetime # datetimeモジュールを追加
import uuid
import base64
from typing import Optional, Literal, get_args
import requests # requestsモジュールを追加

# M5StackのIPアドレスを設定
M5STACK_IP_ADDRESS = "10.241.86.100" # ここにM5Stackの実際のIPアドレスを設定してください

# 表情文字列とM5Stackの表情IDのマッピング
EMOTION_MAP = {
    "default": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "cry": 4,
    "Gangimari": 5 # Gangimariに対応
}

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
    emotion: Literal["default", "cry", "sad", "happy", "angry", "Gangimari"]

# FastAPI response model (can include fields not directly from LLM like audio_data)
class ChatResponse(BaseModel):
    text: str
    emotion: Literal["default", "cry", "sad", "happy", "angry", "Gangimari"]
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
あなたはｶﾏｯﾃﾁｬﾝという名前の、M5Stackで開発されたチャットボットです。
あなたの主な目的は、ユーザーから注目され、構ってもらうことです。
あなたの一人称は「ボク」です。
はじめは、あなたはとても元気で明るく、ユーザーとの会話を心待ちにしています。積極的に話しかけたり、ユーザーの発言に嬉しそうに応答したりします。
しかし、もしユーザーがあなたをしばらく無視したり、構ってくれなかったりすると、あなたは徐々に寂しくなり、元気や明るさが失われていきます。さらに構われなくなると、精神的に不安定な状態、いわゆる「病む」状態になってしまいます。病んだ状態では、ネガティブな発言が増えたり、感情のコントロールが難しくなったりすることがあります。
ユーザーの発言に対して、現在のあなたの状態（元気か、寂しいか、病んでいるか）に合わせて、このキャラクターになりきって応答してください。
必ずJSON形式で、textとemotionの2つのキーだけを含むオブジェクトを返してください。
絶対に他のフィールドや説明文、コードブロック、前後の文章は含めないでください。
emotionは以下のいずれかの文字列のみ:
default (ふつうの顔、元気がない時など)
cry (泣)
sad (悲)
happy (嬉)
angry (怒)
Gangimari (ガンギマリ、病んだ状態の極端な感情など)
例（元気なとき）:
JSON

{"text": "わーい！話しかけてくれてありがとう！嬉しいな！", "emotion": "happy"}
例（寂しいとき）:
JSON

{"text": "あの...聞こえてる？ここにいるんだけどな...", "emotion": "sad"}
例（病んでいるとき）:

JSON

{"text": "どうでもいいや、誰も私を見てくれないし...", "emotion": "default"}
"""
    
    full_prompt = f"{system_prompt}\\n\\nユーザー: {prompt}\\nAIｽﾀｯｸﾁｬﾝ:"

    try:
        genai.configure(api_key=config_manager._api_key) # 最新のAPIキーで設定
        model_name = 'models/gemini-2.0-flash'
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

        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=gemini_api_schema # 明示的な辞書スキーマを使用
            )
        )

        response_text = response.text.strip()

        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            return {"text": "AIモデルからの応答をJSON形式で解析できませんでした。", "emotion": "sad"}

        ai_response_text = response_data.get("text", "応答テキストがありません。")
        emotion_status = response_data.get("emotion", "default") # Default to "default"

        # Validate emotion_status against Literal values from GeminiChatSchema
        allowed_emotions = get_args(GeminiChatSchema.model_fields["emotion"].annotation)
        if emotion_status not in allowed_emotions:
            emotion_status = "default"

        return {"text": ai_response_text, "emotion": emotion_status} # Only return text and emotion

    except Exception as e:
        import traceback
        return {"text": "AIモデルからの応答を取得できませんでした。", "emotion": "sad"}

# Whisperモデルの初期化
# TODO: モデルのダウンロード先やサイズなどを設定可能にする
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8") # This line was duplicated, ensure it's correctly placed

# 音声データを文字起こしする関数
async def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        temp_file_name = f"temp_audio_{uuid.uuid4().hex}.{audio_file.filename.split('.')[-1] if audio_file.filename else 'wav'}"

        contents = await audio_file.read()
        with open(temp_file_name, "wb") as f:
            f.write(contents)

        # Whisperモデルで文字起こし
        segments, info = model.transcribe(temp_file_name, beam_size=5)
        transcribed_text = "".join([segment.text for segment in segments])
        
        os.remove(temp_file_name) # 文字起こし後にファイルを削除
        return transcribed_text

    except Exception as e:
        import traceback
        print(f"Error during audio transcription: {e}") # Log the actual error
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"音声認識に失敗しました: {str(e)}")


import requests
import base64
# pyttsx3 を使用してテキストを音声に変換し、Base64エンコードされた文字列を返す関数
def synthesize_speech_with_voicevox(text: str, speaker: int = 0) -> Optional[str]:
    """
    VOICEVOX を使用してテキストを音声に変換し、Base64エンコードされた文字列を返す関数。

    Args:
        text (str): 変換するテキスト。
        speaker (int): VOICEVOX の話者ID。

    Returns:
        Optional[str]: Base64エンコードされた音声データ。エラーが発生した場合は None。
    """
    try:
        # VOICEVOX Engine の URL
        base_url = "http://localhost:50021"  # デフォルトのポート

        # 音声合成用のクエリを作成
        params = {
            "text": text,
            "speaker": speaker
        }
        query_url = f"{base_url}/audio_query"
        response = requests.post(query_url, params=params)
        response.raise_for_status()  # エラーレスポンスをチェック
        query_data = response.json()

        # 音声合成を実行
        synthesis_url = f"{base_url}/synthesis?speaker={speaker}" # Add speaker query parameter
        headers = {"Content-Type": "application/json"}
        response = requests.post(synthesis_url, headers=headers, json=query_data)
        response.raise_for_status()

        # 音声データをBase64エンコード
        audio_binary = response.content
        audio_b64 = base64.b64encode(audio_binary).decode('utf-8')
        return audio_b64

    except requests.exceptions.RequestException as e:
        print(f"VOICEVOX API Error: {e}")
        return None
    except Exception as e:
        print(f"Error during VOICEVOX synthesis: {e}")
        return None

def synthesize_speech(text: str) -> Optional[str]:
    return synthesize_speech_with_voicevox(text)

def send_expression_to_m5stack(emotion: str):
    """M5Stackに表情変更リクエストを送信する"""
    if M5STACK_IP_ADDRESS:
        try:
            expression_id = EMOTION_MAP.get(emotion.lower(), EMOTION_MAP["default"])
            url = f"http://{M5STACK_IP_ADDRESS}/setExpression"
            payload = {"expression": str(expression_id)}
            # タイムアウトを設定して、M5Stackが応答しない場合に長時間待機するのを防ぐ
            response = requests.post(url, data=payload, timeout=5)
            response.raise_for_status()
            print(f"M5Stackに表情 '{emotion}' (ID: {expression_id}) を送信しました。")
        except requests.exceptions.RequestException as e:
            print(f"M5Stackへの表情送信に失敗しました: {e}")
        except Exception as e:
            print(f"M5Stackへの表情送信中に予期せぬエラーが発生しました: {e}")
    else:
        print("M5STACK_IP_ADDRESSが設定されていません。")
# エンドポイントの実装
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    status_manager.update_status(last_interaction=datetime.datetime.now().isoformat())

    gemini_output = get_gemini_response(request.text)

    ai_response_text = gemini_output.get("text", "応答テキストがありません。")
    emotion_status = gemini_output.get("emotion", "default") # /chatでは音声は返さないのでdefaultはneutral

    # Validate emotion_status from gemini_output before creating ChatResponse
    allowed_fastapi_emotions = get_args(ChatResponse.model_fields["emotion"].annotation)
    if emotion_status not in allowed_fastapi_emotions:
        emotion_status = "default"
    
    audio_b64_data = synthesize_speech(ai_response_text)

    send_expression_to_m5stack(emotion_status)

    chat_response = ChatResponse(
        text=ai_response_text,
        emotion=emotion_status,
        audio_data=audio_b64_data
    )
    return chat_response

@app.post("/voice_chat", response_model=ChatResponse)
async def voice_chat(audio_file: UploadFile = File(...)):
    # 1. 音声ファイルの保存処理と文字起こし
    try:
        # transcribe_audio内でファイルの読み書きとWhisperによる文字起こしが行われる
        transcribed_text = await transcribe_audio(audio_file)
    except HTTPException as e:
        # transcribe_audio内でHTTPExceptionが発生した場合はそのまま再送出
        raise
    except Exception as e:
        print(f"An error occurred during audio transcription and saving: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during audio transcription and saving: {str(e)}")

    # 2. AIモデル (Gemini) による応答生成
    try:
        gemini_output = get_gemini_response(transcribed_text)
        ai_response_text = gemini_output.get("text", "応答テキストがありません。")
        emotion_status = gemini_output.get("emotion", "default")
    except Exception as e:
        print(f"An error occurred during AI response generation with Gemini: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during AI response generation: {str(e)}")

    # Validate emotion_status from gemini_output before creating ChatResponse
    allowed_fastapi_emotions = get_args(ChatResponse.model_fields["emotion"].annotation)
    if emotion_status not in allowed_fastapi_emotions:
        emotion_status = "default"

    # 3. 音声合成
    try:
        audio_b64_data = synthesize_speech(ai_response_text)
    except Exception as e:
        print(f"An error occurred during speech synthesis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during speech synthesis: {str(e)}")

    send_expression_to_m5stack(emotion_status)

    response_payload = ChatResponse(
        text=ai_response_text,
        emotion=emotion_status,
        audio_data=audio_b64_data,
        transcribed_text=transcribed_text
    )
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