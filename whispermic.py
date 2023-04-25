import openai
import os
from io import BytesIO
import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper

# APIキーを設定してください
openai.api_key = ""

def ask_gpt_3_5_turbo(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "元気よく回答してくれるアシスタントAI"}, {"role": "user", "content": f"{text}"}],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].message["content"].strip()
    return message

if __name__ == "__main__":
    model = whisper.load_model("base")

    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone(sample_rate=16_000) as source:
            print("なにか話してください")
            audio = recognizer.listen(source)

        print("音声処理中 ...")
        wav_bytes = audio.get_wav_data()
        wav_stream = BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_fp32 = audio_array.astype(np.float32)

        result = model.transcribe(audio_fp32, fp16=False)
        recognized_text = result["text"]
        print(f"認識されたテキスト: {recognized_text}")

        gpt_response = ask_gpt_3_5_turbo(recognized_text)
        print(f"GPT-3.5-turboの返答: {gpt_response}")
        print("GPT-3.5-turboの返答が終わりました。もう一度話してください。")