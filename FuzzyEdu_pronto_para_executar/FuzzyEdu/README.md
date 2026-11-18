# FuzzyEdu

Aplicação Streamlit para demonstrar inferência fuzzy do tipo Mamdani na avaliação didática de desempenho estudantil.

## Requisitos
- Python 3.10 ou superior
- Conexão com a internet na primeira instalação das bibliotecas

## Windows — executar
1. Extraia os arquivos para uma pasta.
2. Abra o Prompt de Comando nessa pasta.
3. (Recomendado) Crie um ambiente virtual:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Instale as dependências:
   ```bat
   py -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Inicie o aplicativo:
   ```bat
   streamlit run app.py
   ```
6. O navegador abrirá o endereço local indicado no terminal (normalmente http://localhost:8501).

## Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## CSV
Colunas obrigatórias: `Hours_Studied` e `Attendance`.
Coluna opcional: `Exam_Score`.
- Horas: 0 a 50 por semana.
- Frequência: 0 a 100.
- O CSV deve ter cabeçalho; vírgula e ponto e vírgula são aceitos como separadores.

## Observação metodológica
As funções de pertinência e regras são parâmetros didáticos iniciais. Não são um modelo estatístico treinado nem uma previsão validada de nota. Confirme os parâmetros com a especificação da atividade/professor.
