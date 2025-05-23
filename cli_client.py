import requests
import json
import os
import sounddevice as sd
from scipy.io.wavfile import write, read
import wave
import io
import base64
import numpy as np

# バックエンドAPIのエンドポイントURL
# FastAPIアプリケーションがローカルで起動していることを前提とします
API_URL = "http://127.0.0.1:8000"

def chat_with_backend(text: str):
    """
    バックエンドの /chat エンドポイントにリクエストを送信し、応答を表示する
    """
    url = f"{API_URL}/chat"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status() # HTTPエラーが発生した場合に例外を発生させる
        response_data = response.json()

        print("バックエンドからの応答:")
        print(f"  テキスト: {response_data.get('text', 'N/A')}")
        print(f"  表情: {response_data.get('emotion', 'N/A')}")
        if "transcribed_text" in response_data and response_data["transcribed_text"]:
            print(f"  文字起こし: {response_data['transcribed_text']}")
        
        # 音声データの再生を試みる
        if "audio_data" in response_data and response_data["audio_data"]:
            play_audio(response_data["audio_data"])
        # 必要に応じて他のデータも表示

    except requests.exceptions.RequestException as e:
        print(f"バックエンドへのリクエスト中にエラーが発生しました: {e}")
    except json.JSONDecodeError:
        print("バックエンドからの応答が不正なJSON形式です。")

def play_audio(audio_data):
    """
    音声データを再生する
    """
    try:
        # Base64デコード
        decoded_data = base64.b64decode(audio_data)

        # wavデータをNumPy配列に変換
        with io.BytesIO(decoded_data) as f:
            with wave.open(f, 'rb') as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                frames = wf.getnframes()
                data = wf.readframes(frames)
                wf.close()

        # NumPy配列に変換
        audio = np.frombuffer(data, dtype=np.int16)

        # 再生
        sd.play(audio, samplerate=rate) # channels引数を削除
        sd.wait()

    except Exception as e:
        print(f"音声再生中にエラーが発生しました: {e}")

def voice_chat_with_backend(audio_file_path: str):
    """
    バックエンドの /voice_chat エンドポイントに音声ファイルを送信し、応答を表示する
    """
    url = f"{API_URL}/voice_chat"
    try:
        with open(audio_file_path, "rb") as audio_file:
            files = {"audio_file": audio_file}
            response = requests.post(url, files=files)
            response.raise_for_status()
            response_data = response.json()

            print("バックエンドからの応答:")
            if "transcribed_text" in response_data and response_data["transcribed_text"]:
                print(f"  文字起こし: {response_data['transcribed_text']}")
            print(f"  テキスト: {response_data.get('text', 'N/A')}")
            print(f"  表情: {response_data.get('emotion', 'N/A')}")

            # 音声データの再生を試みる
            if "audio_data" in response_data and response_data["audio_data"]:
                play_audio(response_data["audio_data"])

    except FileNotFoundError:
        print(f"エラー: 音声ファイルが見つかりません: {audio_file_path}")
    except requests.exceptions.RequestException as e:
        print(f"バックエンドへのリクエスト中にエラーが発生しました: {e}")
    except json.JSONDecodeError as e:
        print(f"バックエンドからの応答が不正なJSON形式です: {e}")

def record_audio(filename="recording.wav", duration=5, fs=44100, channels=1):
    """
    `sounddevice` を使用して音声を録音し、wavファイルとして保存する
    """
    print(f"録音を開始します... {duration}秒")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels,dtype='int16')
        sd.wait()  # 録音が完了するまで待機
        write(filename, fs, recording)  # Save as WAV file
        print(f"録音完了: {filename}")
        return filename
    except Exception as e:
        import traceback
        print(f"録音中にエラーが発生しました: {e}")
        print(f"エラーの型: {type(e)}")
        print(f"トレースバック: {traceback.format_exc()}")
        return None

def main():
    print("AIｽﾀｯｸﾁｬﾝ バックエンドCLIクライアント")
    print("終了するには 'quit' または 'exit' と入力してください。")

    while True:
        user_input = input("あなた (テキスト, 'record', または音声ファイルのパス): ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input.strip() == "":
            continue

        if user_input.lower() == "record":
            # 録音機能
            recorded_file = None
            try:
                recorded_file = record_audio()
            except Exception as e:
                print(f"録音機能でエラーが発生しました: {e}")

            if recorded_file:
                voice_chat_with_backend(recorded_file)
        elif os.path.exists(user_input):
            # ファイルパスが指定された場合
            voice_chat_with_backend(user_input)
        else:
            # テキストが入力された場合
            chat_with_backend(user_input)

if __name__ == "__main__":
    main()