import time
import torch
import sys
import os
from faster_whisper import WhisperModel

# Verifica argumentos da linha de comando
if len(sys.argv) < 3:
    print("Uso: python script.py <modelo> <arquivo>")
    print("Modelos disponíveis: large-v3, large-v2, medium, small, tiny")
    print("Exemplo: python script.py large-v3 completa.mp3")
    sys.exit(1)

# Obtém parâmetros da linha de comando
modelo = sys.argv[1]
arquivo_entrada = sys.argv[2]

# Verifica se o arquivo existe
if not os.path.exists(arquivo_entrada):
    print(f"Erro: O arquivo '{arquivo_entrada}' não foi encontrado.")
    sys.exit(1)

# Cria nome do arquivo de saída (mesmo nome, extensão .txt)
arquivo_saida = os.path.splitext(arquivo_entrada)[0] + ".txt"

# Detecta GPU e configura parâmetros ideais
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"

# Define workers com base na GPU detectada
num_workers = 4  # Valor padrão
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "4090" in gpu_name: num_workers = 16
    elif "3090" in gpu_name: num_workers = 12
    elif "3080" in gpu_name: num_workers = 6
    elif "3070" in gpu_name: num_workers = 4
    print(f"GPU: {gpu_name} | Workers: {num_workers}")
else:
    print("Nenhuma GPU detectada. Usando CPU (será mais lento).")

# Carrega o modelo
print(f"Carregando modelo {modelo}...")
model = WhisperModel(
    modelo, 
    device=device, 
    compute_type=compute_type,
    num_workers=num_workers
)

# Transcreve o arquivo
print(f"Iniciando transcrição de: {arquivo_entrada}")
start_time = time.time()

# Executa a transcrição com parâmetros otimizados
segments, info = model.transcribe(
    arquivo_entrada,

# Define explicitamente o idioma da transcrição como português
    language='pt',

# Usa busca em feixe (beam search) com 5 caminhos, o que melhora a qualidade da transcrição
# comparado à busca gulosa (padrao = 1), mas aumenta um pouco o tempo de processamento
    beam_size=5,

# Faz com que o modelo considere o texto anterior no momento de gerar a próxima parte
# Isso melhora a fluidez e consistência em conversas longas ou emocionais
    condition_on_previous_text=False,

# Ativa o filtro VAD (Voice Activity Detection) para eliminar segmentos de silêncio ou ruído
    vad_filter=True,

# Usa temperatura 0.0 para deixar o modelo o mais determinístico possível
# Isso ajuda a reduzir "alucinações" e gerar transcrições mais fiéis ao áudio real
    temperature=0.0,

# Define a tarefa como "transcrição" (e não tradução)
    task='transcribe',

# Retorna os timestamps de cada palavra individualmente
# Essencial se você pretende fazer análise mais avançada como
# diarização com pyannote, marcação por palavra ou geração de highlights
#    word_timestamps=True,

# Define o limite de detecção de ausência de fala (silêncio) como mais agressivo
# Reduz ainda mais a chance de captar ruídos ou palavras ditas em sussurro fora de contexto
    no_speech_threshold=0.7

#    best_of=3,
#    word_timestamps=True
)

# Processa os segmentos e imprime enquanto salva
with open(arquivo_saida, "w", encoding="utf-8") as f:
    print("\n--- TRANSCRIÇÃO EM ANDAMENTO ---\n")
    for segment in segments:
        timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
        text = segment.text
        
        # Imprime no terminal
        print(f"{timestamp} {text}")
        
        # Salva no arquivo
        f.write(f"{timestamp} {text}\n")

# Estatísticas finais
elapsed = time.time() - start_time
rtf = elapsed / info.duration if info.duration > 0 else 0
print(f"\n--- ESTATÍSTICAS ---")
print(f"Transcrição concluída em {elapsed:.2f}s")
print(f"Duração do áudio: {info.duration:.2f}s")
print(f"RTF: {rtf:.2f}x (velocidade de {1/rtf:.2f}x tempo real)")
print(f"Resultado salvo em: {arquivo_saida}")
