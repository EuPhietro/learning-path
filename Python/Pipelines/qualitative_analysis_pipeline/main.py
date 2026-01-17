import os
import re
import json
import pandas as pd
from groq import Groq
from time import sleep
from dotenv import load_dotenv

"""
# Pipeline de Análise Qualitativa Automatizada com IA

## Visão Geral
Este projeto implementa um pipeline de dados para processar e enriquecer pesquisas qualitativas. Utiliza Inteligência Artificial Generativa (LLM) para analisar respostas em texto aberto e categorizá-las automaticamente com base em uma taxonomia pré-definida, garantindo padronização para análises quantitativas posteriores.

## Stack Tecnológico (SDKs)
* **Groq SDK:** Abstração e comunicação de alta performance com a API de inferência de IA.
* **Pandas SDK:** Manipulação, estruturação e consolidação de dados tabulares (Excel/DataFrames).

## Especificações do Modelo de IA
O modelo selecionado foi o **`meta-llama/llama-4-scout-17b-16e-instruct`**, escolhido pelo equilíbrio entre capacidade de instrução e limites operacionais da API:
* **RPM (Requisições/min):** 30
* **RPD (Requisições/dia):** 1K
* **TPM (Tokens/min):** 30K
* **TPD (Tokens/dia):** 500K

## Configuração e Constantes
O comportamento do pipeline é regido por constantes que definem o contexto e as restrições do modelo de IA:

* **`MAP_COLS`**: Dicionário usado para renomear as chaves curtas do JSON de saída da IA para nomes de colunas legíveis no DataFrame final consolidado.
* **`CATEGORIAS_PERMITIDAS`**: Lista fechada de categorias alvo da pesquisa.
    * *Justificativa:* As categorias foram definidas arbitrariamente com base no escopo restrito da pesquisa. O uso de uma lista fechada no prompt é crucial para evitar que a IA gere categorias homônimas ou sinônimas (ex: criar "Gestão de Pessoas" quando já existe "Liderança"), garantindo a integridade dos dados agrupados.
* **`SCORE_SENTIMENTOS`**: Escala de Likert (5 pontos) usada para mensurar o índice de satisfação ou insatisfação em cada resposta categorizada.

## Estrutura da Tabela Final
Ao final do processamento, é gerada uma tabela plana (flat table) contendo as informações tratadas e enriquecidas.

* **Granularidade:** O pipeline é capaz de fragmentar uma única resposta original (uma célula do Excel) em múltiplos registros na tabela final, caso o comentário aborde diferentes temas (ex: um feedback que mencione simultaneamente "gestão" e "performance").
* **Integridade:** A tabela final mantém a conexão com os dados originais, preservando a subjetividade e o comentário real da pesquisa para auditorias ou análises mais profundas.

## Aplicabilidade e Casos de Uso
A aplicabilidade deste projeto é ampla, podendo ser voltada para:
* **Análise de NPS e Satisfação:** Categorização e enriquecimento de dados de perguntas abertas em pesquisas de satisfação do cliente (CSAT, NPS) provenientes de formulários web ou apps móveis.
* **Mapeamento de Comportamento:** Identificação de padrões de comportamento e dores dos usuários através da análise massiva de feedbacks qualitativos.
"""


# CARREGANDO NOSSA CHAVE API DAS VARIÁVEIS DE AMBIENTE
load_dotenv()  

# Configurações Globais
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# DECLARAÇÃO DAS NOSSAS VARIÁVEIS CONSTANTES

MAP_COLS = {
    "cat": "Categoria",
    "sum": "Resumo_Executivo",
    "prob": "Ponto_Critico",
    "sent": "Sentimento",
    "sols": "Sugestoes_de_Melhoria",
    "tags": "Palavras_Chave"
}

MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'

CATEGORIAS_PERMITIDAS = ['Cultura e Reconhecimento','Capacitação Técnica', 'Liderança e Proximidade','Eficiência Operacional', 'Infraestrutura e Recursos', 'Comunicação e Alinhamento', 'Geral/Outros']

SCORE_SENTIMENTOS = ['Feliz','Pouco Feliz', 'Neutro','Pouco Triste', 'Infeliz']

# DECLARAÇÃO DA FUNÇÃO USADA PARA PROCESSAR AS ENTRADAS

def process_text(prompt: str) -> str | None:
    """
    Processa um texto utilizando um modelo de IA para extrair e categorizar informações qualitativas.

    Esta função envia o texto fornecido (prompt) para um modelo de linguagem da Groq, configurado com instruções de sistema específicas para atuar como um Analista Qualitativo. O modelo é instruído a extrair insights, dividir mensagens longas em temas e retornar os dados em um formato JSON padronizado e estrito.

    Args:
        prompt (str): O texto bruto (comentário, feedback) que será analisado pela IA.

    Returns:
        str | None: Uma string contendo um array JSON de objetos, onde cada objeto representa uma análise categorizada.
                    Retorna `None` em caso de erro na comunicação com a API.

    Formato do JSON de Retorno (Array de Objetos):
        [
          {
            "cat": "Área temática (conforme CATEGORIAS_PERMITIDAS)",
            "sum": "Resumo conciso do ponto abordado",
            "prob": "Descrição do ponto crítico ou problema identificado",
            "sent": "Análise de sentimento (conforme SCORE_SENTIMENTOS)",
            "sols": ["Lista de sugestões de melhoria"],
            "tags": ["Lista de até 3 palavras-chave"]
          },
          ... (pode conter múltiplos objetos se o prompt abordar vários temas)
        ]

    Raises:
        Exception: Captura e imprime qualquer exceção que ocorra durante a chamada à API da Groq (ex: erro de rede, timeout, erro de autenticação), retornando `None`.
    """
    # Use {{ para o que for texto JSON e { para variáveis Python
    system_instructions = f"""
    Analista Qualitativo. Extraia dados em JSON (ARRAY de objetos).
    Divida mensagens grandes e categorize-as por partes/temas diferentes.
    Esquema esperado:
    [{{
        "cat": "Área temática",
        "sum": "Resumo",
        "prob": "Ponto crítico",
        "sent": "{SCORE_SENTIMENTOS}",
        "sols": ["Sugestões"],
        "tags": ["3 keywords"]
    }}]
    REGRAS:
    1. Retorne APENAS o JSON puro (Array de objetos).
    2. Use APENAS estas categorias: {CATEGORIAS_PERMITIDAS}
    3. Use as chaves curtas: cat, sum, prob, sent, sols, tags.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": 'json_object'}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erro na chamada da API: {e}")
        return None
    



def to_table(json_string: str | None) -> pd.DataFrame | None:
    """
    Converte uma string JSON bruta em um pandas DataFrame, com normalização de estrutura.

    Esta função é projetada para processar o retorno textual de um modelo de IA e transformá-lo em uma estrutura tabular robusta. Ela atua como uma camada de resiliência, lidando com as variações comuns no formato como o JSON pode ser retornado pela IA (como um objeto único ou uma lista de objetos).

    A função normaliza a entrada para garantir que o construtor do pandas (`pd.DataFrame`) receba sempre uma **lista de dicionários**. Esse passo é crítico para evitar o erro "All arrays must be of the same length", que ocorre quando o pandas tenta interpretar um único dicionário contendo listas internas de tamanhos variados como colunas, em vez de uma única linha.

    Args:
        json_string (str | None): A string formatada em JSON para ser processada.
                                  Se for `None` ou uma string vazia, a função retorna `None` imediatamente.

    Returns:
        pd.DataFrame | None: Um DataFrame do pandas onde cada objeto do JSON original se torna uma linha.
                             Retorna `None` se a entrada for vazia, se ocorrer um erro de parsing do JSON (json.JSONDecodeError), ou se a estrutura dos dados não for compatível (nem lista nem ditado).

    Comportamento de Normalização (Cenários Tratados):
    --------------------------------------------------
    1.  **Lista Direta (Array de Objetos):**
        - Entrada: `[{"id":1, "tags":["a","b"]}, {"id":2, "tags":["c"]}]`
        - Ação: Nenhuma alteração necessária. O pandas cria múltiplas linhas.

    2.  **Dicionário "Wrapper" (Chave Mestre):**
        - Entrada: `{"resultados": [{"id":1}, {"id":2}]}`
        - Ação: Identifica que há apenas uma chave contendo uma lista e extrai essa lista interna para processamento.

    3.  **Objeto Único:**
        - Entrada: `{"id":1, "tags":["a", "b", "c"]}`
        - Ação: Envolve o objeto em uma lista: `[{"id":1, "tags":["a", "b", "c"]}]`. Isso força o pandas a criar uma única linha, mantendo a lista interna `["a", "b", "c"]` intacta dentro de uma única célula, evitando erros de alinhamento.

    Raises:
        Captura qualquer `Exception` genérica durante o parsing ou conversão (como `json.JSONDecodeError` ou `TypeError`) e imprime um erro técnico no console antes de retornar `None`.
    """
    if not json_string:
        return None

    try:
        data = json.loads(json_string)

        # CENÁRIO 1: A IA retornou uma lista direta [{}, {}]
        if isinstance(data, list):
            # Se for uma lista de dicionários, o Pandas lida bem e cria múltiplas linhas.
            pass

        # CENÁRIO 2: A IA retornou um dicionário (objeto único ou wrapper)
        elif isinstance(data, dict):
            # Sub-cenário A: Dicionário wrapper {"resultados": [{}, {}]}
            # Se o dicionário tem apenas 1 chave e o valor dessa chave é uma lista,
            # assumimos que é um "wrapper" e extraímos a lista interna.
            if len(data) == 1 and isinstance(list(data.values())[0], list):
                data = list(data.values())[0]

            # Sub-cenário B: Um único objeto de dados {...}
            # Envolvemos este objeto em uma lista [{}]. Isso é crucial para que o pandas
            # trate este objeto como UMA linha, e não tente interpretar suas chaves como índice
            # e suas listas internas como colunas (o que causaria erro de tamanho de array).
            else:
                data = [data]

        # Se não for nem lista nem dicionário após o loads (ex: um int ou string solta), não é válido.
        else:
            return None

        # Criação do DataFrame
        # O Pandas mapeia cada dicionário da lista final para uma linha da tabela
        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f'Erro técnico na conversão JSON: {e}')
        return None
    




def main():
    """
    Função principal de orquestração do pipeline de análise qualitativa.

    Esta função coordena todo o fluxo de trabalho ETL (Extração, Transformação e Carga) do projeto.
    Ela é responsável por ler os dados brutos, iterar sobre as perguntas e respostas, gerenciar a comunicação com a IA, enriquecer os resultados com metadados para rastreabilidade e consolidar tudo em um relatório final.

    Fluxo de Execução:
    ------------------
    1.  **Ingestão de Dados:** Tenta carregar o arquivo Excel de origem ('Pesquisa de melhoria AHS(1-24).xlsx'). Se falhar, encerra a execução.
    2.  **Filtragem de Colunas:** Identifica dinamicamente quais colunas contêm as perguntas abertas para análise, ignorando colunas administrativas (como 'Start time', 'Email') e separando a coluna agrupadora ('Selecione o grupo...').
    3.  **Processamento Iterativo (Loop Aninhado):**
        - Percorre cada coluna de pergunta identificada.
        - Percorre cada linha (respondente) da planilha original.
        - Verifica se o feedback é válido (não vazio/NaN).
        - Chama `process_text()` para obter a análise da IA.
        - Chama `to_table()` para converter a resposta da IA em um DataFrame temporário.
    4.  **Enriquecimento de Dados (Metadata Injection):**
        - Adiciona colunas cruciais ao DataFrame temporário para garantir a rastreabilidade:
            - `id_original`: O ID da linha na planilha mãe.
            - `grupo`: O grupo demográfico do respondente.
            - `pergunta_original`: O texto da pergunta que gerou aquele feedback.
            - `feedback_bruto`: O texto original escrito pelo respondente.
    5.  **Controle de Taxa (Rate Limiting):** Aplica uma pausa de 4 segundos (`sleep(4)`) após cada chamada bem-sucedida para respeitar os limites operacionais da API da Groq (evitando erros 429).
    6.  **Consolidação e Exportação:**
        - Concatena todos os DataFrames temporários acumulados na lista `master_results` em um único "Master DataFrame".
        - Renomeia as colunas usando a constante global `MAP_COLS` para nomes mais legíveis.
        - Reorganiza a ordem das colunas, priorizando os metadados de contexto à esquerda e a análise da IA à direita.
        - Salva o resultado final no arquivo 'Analise_Qualitativa_Consolidada.xlsx'.

    Dependências Globais:
        - Requer a existência do arquivo 'Pesquisa de melhoria AHS(1-24).xlsx' no diretório de execução.
        - Depende da constante global `MAP_COLS` para a renomeação final das colunas.
        - Depende das funções auxiliares `process_text` e `to_table`.

    Efeitos Colaterais:
        - Lê um arquivo do disco.
        - Imprime logs de progresso no console padrão (stdout).
        - Pausa a execução do script (sleep).
        - Cria ou sobrescreve um arquivo Excel no disco ('Analise_Qualitativa_Consolidada.xlsx').

    Returns:
        None.
    """
    file_name = os.environ.get("DATABASE")
    try:
        df_pesquisa = pd.read_excel(file_name)
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        return

    # 1. Identificação das colunas de interesse
    cols_ignore = ['Start time', 'Completion time', 'Email', 'Name', 'Last modified time', 'ID']
    grupo_col = 'Selecione o grupo a qual você pertence'
    
    # Perguntas são todas as colunas que não estão no ignore e não são a de grupo
    perguntas_cols = [c for c in df_pesquisa.columns if c not in cols_ignore and c != grupo_col]

    master_results = []

    # 2. Processamento
    for col in perguntas_cols:
        print(f'\n🔍 Analisando Pergunta: "{col}"')
        
        for index, row in df_pesquisa.iterrows():
            feedback = str(row[col])
            id_origem = row['ID'] if 'ID' in row else index
            grupo_origem = row[grupo_col] if grupo_col in df_pesquisa.columns else "N/A"

            if feedback.strip() and feedback.lower() != 'nan':
                raw_json = process_text(feedback)
                df_temp = to_table(raw_json)

                if df_temp is not None:
                    # Injeção de Metadados para rastreabilidade
                    df_temp['id_original'] = id_origem
                    df_temp['grupo'] = grupo_origem
                    df_temp['pergunta_original'] = col
                    df_temp['feedback_bruto'] = feedback
                    
                    master_results.append(df_temp)
                    print(f'✅ ID {id_origem} processado.')
                    
                    # Cota: 30 RPM -> 1 req/2s. Usamos 4s para segurança total.
                    sleep(4)

    # 3. Consolidação Final
    if master_results:
        df_final = pd.concat(master_results, ignore_index=True)
        
        # Renomeação profissional
        df_final = df_final.rename(columns=MAP_COLS)

        # Reorganização das colunas (Metadados à esquerda, Análise à direita)
        cols_ordem = ['id_original', 'grupo', 'pergunta_original', 'Categoria', 
                      'Sentimento', 'Resumo_Executivo', 'Ponto_Critico', 
                      'Sugestoes_de_Melhoria', 'Palavras_Chave', 'feedback_bruto']
        
        # Filtra apenas as colunas que realmente existem
        df_final = df_final[[c for c in cols_ordem if c in df_final.columns]]

        df_final.to_excel("Analise_Qualitativa_Consolidada.xlsx", index=False)
        print(f"\n🏆 SUCESSO! {len(df_final)} análises consolidadas em 'Analise_Qualitativa_Consolidada.xlsx'")
    else:
        print("Nenhum dado processado.")

if __name__ == "__main__":
    main()