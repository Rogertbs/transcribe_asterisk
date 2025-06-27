#!/usr/bin/env python3

import paramiko
from scp import SCPClient
import asyncio
import websockets
import io
import requests
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
from datetime import datetime

# Inicializa o modelo (use 'cuda' se quiser GPU)
model = WhisperModel("large-v3", compute_type="float16", device="cuda")
#model = WhisperModel("large-v3", compute_type="int8", device="cpu")

# Parâmetros de áudio
SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (16-bit PCM)
CHUNK_DURATION_SECONDS = 10
CHUNK_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_SECONDS

async def handler(websocket):
    print("📞 Nova conexão de áudio do Asterisk")

    buffer = bytearray()
    segment_count = 0

    try:
        async for message in websocket:
            buffer.extend(message)

            while len(buffer) >= CHUNK_SIZE:
                # Extrai um chunk de áudio do buffer
                audio_chunk = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]

                # Salva em WAV para transcrição
                filename = f"chunk_{datetime.now().strftime('%H%M%S_%f')}.wav"
                print(filename)
                with sf.SoundFile(filename, mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS,
                                  format='WAV', subtype='PCM_16') as f:
                    data = np.frombuffer(audio_chunk, dtype='int16')
                    f.write(data)

                # Transcreve
                segments, _ = model.transcribe(filename, language='pt')
                print("📝 Transcrição parcial:")
                for seg in segments:
                    print(f"{seg.start:.1f}s - {seg.end:.1f}s: {seg.text}")
                     
                    # Envia ao Ollama e recebe resposta
                    return_ia = await send_ia(seg.text)
                    # Envia resposta Ollama para GPT Gerar TTS
                    teste = await generate_file_audio(str(return_ia))
                    print("🤖 Resposta da IA:", return_ia)

                    # Aqui você pode salvar em um arquivo para o Asterisk ler, ou gerar TTS direto
                    with open("/tmp/resposta_ollama.txt", "w") as f:
                        f.write(return_ia)

    except websockets.exceptions.ConnectionClosed:
        print("🔚 Conexão encerrada.")


async def send_ia(prompt):
    cmd = "Prompt:Obs: Sempre de Respostar Curtas, curtissimas. [O seu nome é Geraldina. De respostas curtas e objetivas de forma educada, como se você fosse um atendente de plano de saúde, e gostaria de ajudar o paciente a agendar consultas.] Pergunta: "
    cmd+= prompt
    payload = {
        "model": "llama3.1:latest",  # substitua pelo modelo que está rodando no Ollama
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        resposta = response.json()["response"]
        return resposta.strip()
    except Exception as e:
        print("Erro ao consultar o Ollama:", e)
        return "Desculpe, não consegui responder."


async def generate_file_audio(response_ia):
    from pathlib import Path
    from openai import OpenAI

    client = OpenAI()
    speech_file_path = Path(__file__).parent / "speech.mp3"
    speech_file_wav = Path(__file__).parent / "speech.wav"

    print("Tentando gerar audio")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=response_ia,
    ) as response:
        response.stream_to_file(speech_file_path)
        
        audio = AudioSegment.from_mp3(speech_file_path)
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        audio.export(speech_file_wav, format="wav")

        send_audio = await send_audio_to_asterisk("/opt/rogerio-lab/speech.wav")


async def send_audio_to_asterisk(file_local, destination="/var/lib/asterisk/sounds/custons/", host="200.152.191.137", usuario="root"):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=usuario, key_filename="/root/.ssh/id_rsa")

        with SCPClient(ssh.get_transport()) as scp:
            scp.put(file_local, destination)
            print("✅ Arquivo enviado com sucesso.")
        
            play_audio = await play_audio_channel_asterisk()

    except Exception as e:
        print(f"❌ Falha ao enviar arquivo: {e}")


async def play_audio_channel_asterisk():
    # Pega id do canal
    # Configurações da API ARI
    channel = await get_channel()
    print(channel)

    # Dados da conexão
    url = f"http://200.152.191.137:5088/asterisk/ari/channels/{channel}/play"
    usuario = "asterisk"
    senha = "asterisk"

    # Parâmetros da requisição
    params = {
        "media": "sound:custons/speech"  # Verifique se é 'custons' ou 'customs'
    }

    # Fazendo a requisição POST para o ARI
    response = requests.post(url, auth=(usuario, senha), params=params)

    # Verificando a resposta
    if response.status_code == 200:
        print("✅ Áudio tocando no canal com sucesso.")
    else:
        print(f"❌ Erro ao tocar áudio: {response.status_code}")
        print(response.text)

async def get_channel():
    # Configurações da API ARI
    url = "http://200.152.191.137:5088/asterisk/ari/channels"
    usuario = "asterisk"
    senha = "asterisk"

    response = requests.get(url, auth=(usuario, senha))

    if response.status_code == 200:
        canais = response.json()
        if len(canais) == 0:
            print("❌ Nenhum canal encontrado.")
        elif len(canais) == 1:
            channel_id = canais[0]['id']
            print(f"✅ Canal encontrado. Channel ID: {channel_id}")
            return channel_id
        else:
            print("⚠️ Existem múltiplos canais ativos:")
            for canal in canais:
                print(f"- ID: {canal['id']}, Name: {canal['name']}")
    else:
        print(f"❌ Erro ao consultar canais. Status: {response.status_code}")
        print(response.text)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("🔊 Servidor WebSocket de transcrição iniciado em ws://0.0.0.0:8765")
        await asyncio.Future()  # Mantém o servidor rodando para sempre

if __name__ == "__main__":
    asyncio.run(main())

