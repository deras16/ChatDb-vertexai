from dotenv import load_dotenv
# from feedback_chat import ChatEvaluationStorage
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import bigquery
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
import streamlit as st
from streamlit_feedback import streamlit_feedback
import os
import pandas as pd
from typing import Optional
import time

load_dotenv()

# Configuración de credenciales
google_credentials_dir = os.getenv("GOOGLE_CLOUD_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_dir

def init_bigquery_client():
    try:
        client = bigquery.Client()
        return client
    except Exception as e:
        st.error(f"Error initializing BigQuery client: {str(e)}")
        return None

if "client" not in st.session_state:
    st.session_state.client = init_bigquery_client()
if "feedback_responses" not in st.session_state:
    st.session_state.feedback_responses = {}

def handle_feedback(feedback, response_id):
    """Procesa y almacena el feedback recibido"""
    if feedback:
        st.session_state.feedback_responses[response_id] = feedback
        #se extrae el id del response_id
        interaction_id = int(response_id.split('_')[1])

        st.toast("✔️ ¡Gracias por tu feedback!")


def validate_sql_query(query: str) -> tuple[bool, str]:
    required_keywords = ['SELECT', 'FROM']
    query_upper = query.upper()
    
    if not all(keyword in query_upper for keyword in required_keywords):
        return False, "La consulta SQL generada no es válida"
    
    if len(query.strip()) < 10:
        return False, "La consulta SQL es demasiado corta"
        
    return True, ""

def get_schema(client):
    try:
        query = """
            SELECT 
            table_name, 
            column_name, 
            data_type 
            FROM ai-mag-431021.CHATBOT.INFORMATION_SCHEMA.COLUMNS
            ORDER BY table_name, ordinal_position
        """
        results = client.query(query).result()
        
        schema_info = []
        for row in results:
            schema_info.append(f"Table: {row.table_name}, Column: {row.column_name}, Type: {row.data_type}")
        
        return "\n".join(schema_info)
    except Exception as e:
        return f"Error getting schema: {str(e)}"

def get_bigquery_chain(client, dataset_id):
    template = """
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

      <SCHEMA>{schema}</SCHEMA>

      All table names must be fully qualified with the dataset name {dataset_id}. 
      
      Important rules for SQL generation:
      1. Always use proper table qualification with dataset_id
      2. All string values in WHERE clauses must be UPPER case and not NOT contain any accent marks.
      3. Ensure all column names exactly match the schema
      4. For complex queries, break them down into CTEs (Common Table Expressions)
      5. Include appropriate aggregation functions when needed
      6. Add proper GROUP BY clauses when using aggregations
      7. Always test for NULL values when necessary
      8. Use SAFE_DIVIDE for division operations
      
      Remember: 
      - The table granosbasicos stores information about corn, beans, sorghum
      - The table hortalizas contains information related to vegetables

      Conversation History: {chat_history}
      
      Write only the SQL query and nothing else. Do not include any explanation or markdown formatting.
      For example:
      Question: what was the corn production in the department of chalatenango in the season of apante in the year 2022 ?
      SQL Query: SELECT SUM(PRODUCCION) PRODUCTION FROM `Chatbot.granosbasicos`
      WHERE GRANO = UPPER('maiz') AND DEPTO = UPPER('chalatenango') AND EPOCA = UPPER('invierno') AND ANIO = 2022;
      Question: What are the 3 grains with highest production by department in the year 2022?
      SQL Query: WITH MaxProduction AS(
        SELECT GRANO, DEPTO, SUM(PRODUCCION) PRODUCTION, 
        ROW_NUMBER() OVER(PARTITION  BY GRANO ORDER BY SUM(PRODUCCION) DESC) AS RN
        FROM CHATBOT.granosbasicos
        WHERE ANIO = 2022
        GROUP BY GRANO, DEPTO
      )
      SELECT MP.GRANO, MP.DEPTO, MP.PRODUCTION 
      FROM MaxProduction MP
      WHERE MP.RN = 1 
      ORDER BY MP.PRODUCTION DESC
      LIMIT 3;
      Question: Which seed has the best yield for corn in the year 2022?
      SQL Query:WITH cterendimiento AS(
        SELECT r.TRANSACTIONID, SAFE_DIVIDE(r.PRODUCCION,r.SUPERFICIE) RENDIMIENTO 
        FROM CHATBOT.granosbasicos AS r
      ), rendimientosemilla AS(
        SELECT cb.GRANO, cb.SEMILLA, cb.DEPTO, AVG(ct.RENDIMIENTO) RENDIMIENTO FROM CHATBOT.granosbasicos cb
        INNER JOIN cterendimiento ct ON ct.TRANSACTIONID = cb.TRANSACTIONID
        WHERE cb.ANIO = 2022 AND cb.GRANO = UPPER('maiz')
        GROUP BY cb.GRANO, cb.SEMILLA, cb.DEPTO 
      ), MAXRENDIMIENTO AS (
        SELECT rs.DEPTO,MAX(rs.RENDIMIENTO) maxrendimiento FROM rendimientosemilla rs
        GROUP BY rs.DEPTO
      )
      SELECT rs.grano, rs.semilla, rs.depto, rs.rendimiento 
      FROM rendimientosemilla rs
      INNER JOIN MAXRENDIMIENTO mx ON mx.depto = rs.depto and mx.maxrendimiento = rs.RENDIMIENTO
      Question: What was the production of cucumber in 2022?
      SQL Query: SELECT SUM(PRODUCCION) as Production FROM `Chatbot.hortalizas`
      WHERE NombreHortaliza = UPPER('pepino') AND ANIO = 2022;
      Question: What was the structure with the highest production in 2022?
      SQL Query: SELECT ESTRUCTURA,SUM(PRODUCCION) AS PRODUCTION 
      FROM `Chatbot.hortalizas`
      WHERE ANIO = 2022 AND ESTRUCTURA != 'CAMPO ABIERTO'
      GROUP BY ESTRUCTURA
      ORDER BY PRODUCTION
      DESC LIMIT 1;

      Your turn:

      Question: {question}
      SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatVertexAI(
        model="gemini-1.5-pro-001",
        max_output_tokens=2048,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        max_retries=3
    )

    return (
        RunnablePassthrough.assign(schema=lambda _: get_schema(client))
        | prompt
        | llm
        | StrOutputParser()
    )
    #funcion modificada, solo tenia la limpieza
def exec_query(client, query: str) -> Optional[pd.DataFrame]:
    try:
        # Limpiar y validar la consulta, quitar estilo markdown
        clean_query = query.replace('```sql', '').replace('```', '').strip()
        is_valid, error_message = validate_sql_query(clean_query)
        
        if not is_valid:
            return pd.DataFrame({'error': [error_message]})
            
        # Ejecutar sin timeout en la configuración
        query_job = client.query(clean_query)
        
        # Esperar resultado con timeout manual
        start_time = time.time()
        max_wait_seconds = 60  # Aumentamos el tiempo máximo de espera a 60 segundos
        
        while True:
            try:
                # Intentar obtener el resultado con un timeout de espera
                df = query_job.result(timeout=30).to_dataframe()
                if df.empty:
                    return pd.DataFrame({'error': ["La consulta no retornó resultados"]})
                return df
            except Exception as e:
                if time.time() - start_time > max_wait_seconds:
                    return pd.DataFrame({'error': ["La consulta excedió el tiempo máximo de espera"]})
                time.sleep(1)
                continue
                
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            return pd.DataFrame({'error': ["La consulta está tomando demasiado tiempo. Por favor, intenta una consulta más específica."]})
        return pd.DataFrame({'error': [f"Error al ejecutar la consulta: {error_msg}"]})

def get_response(user_query: str, client, chat_history: list, dataset_id: str):
    sql_chain = get_bigquery_chain(client, dataset_id)

    template = """ 
        You are a data analyst at a company. Based on the SQL query results, provide a clear and complete answer to the user's question.
        Important formatting rules:
        1. Use clear paragraphs with proper spacing
        2. For lists, use consistent numbering and proper spacing
        3. For production quantities, use format: "1,234.56 quintales"
        4. For areas, use format: "1,234.56 manzanas"
        5. Use proper punctuation and spacing
        6. Avoid using markdown formatting like ** or *
        7. Present numbered lists in a clean format with one item per line
        8. Present statistics in a clear, easy-to-read format
        
        Format example for lists:
        Los departamentos con mayor producción son:
        1. San Salvador: 1,234.56 quintals
        2. La Libertad: 987.45 quintals
        3. Santa Ana: 654.32 quintals

        Format example for statistics:
        La producción total fue de 1,234.56 quintals, con un rendimiento promedio de 45.67 quintals por manzana.
        
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        
        Provide a complete and well-structured answer using the formatting rules above:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatVertexAI(
        model="gemini-1.5-pro-001",
        max_output_tokens=2048,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        max_retries=3
    )

    def process_query(vars):
       try:
        print("Generated SQL Query:", vars["query"])
        result = exec_query(client, vars["query"])
        if 'error' in result.columns:
            # Si hay un error, formatear el mensaje para el LLM
            error_message = result['error'].iloc[0]
            return f"Error en la consulta: {error_message}"
        return result
       except Exception as e:
        return f"Error inesperado: {str(e)}"
    
    def clean_response(text: str) -> str:
        # Dividir por oraciones primero
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        cleaned_sentences = []
        
        for sentence in sentences:
            # Si es un elemento numerado de lista
            if sentence.strip() and sentence[0].isdigit():
                # Asegurar que cada elemento numerado empiece en nueva línea
                cleaned_sentences.append('\n' + sentence.strip())
            else:
                cleaned_sentences.append(sentence.strip())
        
        # Unir las oraciones
        cleaned = '. '.join(cleaned_sentences)
        
        # Asegurar salto de línea después de "siguiente:" o similar
        indicators = ['siguiente:', 'son:', 'fueron:', 'siguientes:']
        for indicator in indicators:
            if indicator in cleaned.lower():
                cleaned = cleaned.replace(indicator, indicator + '\n')
        
        # Remover espacios múltiples
        cleaned = ' '.join(cleaned.split())
        
        # Asegurar que no hay espacios extras antes de los dos puntos
        cleaned = cleaned.replace(' :', ':')
        
        # Asegurar espacio después de los dos puntos
        cleaned = cleaned.replace(':', ': ')
        
        return cleaned.strip()
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: client.list_tables(dataset_id),
            response=lambda vars: process_query(vars)
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response_stream = chain.stream({
            "question": user_query,
            "chat_history": chat_history,
            "dataset_id": dataset_id,
        })
        # Acumular y limpiar la respuesta antes de mostrarla
        full_response = ""
        for chunk in response_stream:
            full_response += chunk
        
        # Limpiar y formatear la respuesta completa
        cleaned_response = clean_response(full_response)
        
        # Devolver la respuesta limpia como un iterador
        return iter([cleaned_response])
    
    except Exception as e:
        return iter([f"Lo siento, hubo un problema al procesar tu pregunta. Por favor, intenta hacerla de manera más específica o reformúlala."])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola soy Asistente OIAD, un chatbot para solventar dudas de siembra y producción de granos basicos y hortalizas. "
                 "Recuerda que me encuentro en una etapa de desarrollo, la información que genero no es oficial y debe ser validada."),
    ]

def main():
    load_dotenv()

    st.set_page_config(
        page_title="Asistente MAG-OIAD",
        page_icon="config\gobierno.png",
        layout="wide",
    )
    
    col1, col2 = st.columns([8, 1])
    with col1:
        st.title("Asistente MAG-OIAD")
    with col2:
        st.image("config\MAG.png")

    st.markdown("""
        <style>
            .st-emotion-cache-czk5ss.e16jpq800 {
                visibility: hidden;
            }
            .stDeployButton {
                visibility: hidden;
            }
            .st-emotion-cache-bm2z3a ea3mdgi8 {
                background-color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Settings")
        st.write("Esto es una prueba de un chat con BigQuery usando vertexAI")

        with st.expander("Ejemplos de prompts", expanded=True):
            st.write(
            """
                - Cuentame sobre la información que me puedes proporcionar para granosbasicos y hortalizas.
                - ¿Cual es la produccion de maiz en el año 2022?
                - Quiero saber los 5 departamentos con mayor cultivo de tomate
                - Cual es la estructura que genero una mayor produccion de tomate en el año 2019? 
                - Cuanto es la superficie para sorgo en la epoca de apante en el año 2019?
            """
            )
        st.text_input("GOOGLE PROJECTID", value="AI-MAG", key="Host", disabled=True)
        st.text_input("DATASET", value="CHATBOT", key="dataset", disabled=True)

    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
                if i > 0:
                    response_id = f"feedback_{i}"
                    if response_id not in st.session_state.feedback_responses:
                        streamlit_feedback(
                            feedback_type="thumbs",
                            key=response_id,
                            align="flex-start",
                            on_submit=lambda feedback, rid=response_id: handle_feedback(feedback, rid)
                        )
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Type a message...")
    dataset_id = "ai-mag-431021.Chatbot"
    
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            try:
                response = get_response(user_query, st.session_state.client, st.session_state.chat_history, dataset_id)
                response_text = ""
                for chunk in response:
                    response_text += chunk
                    st.write(chunk)
                st.session_state.chat_history.append(AIMessage(content=response_text))

                response_id = f"feedback_{len(st.session_state.chat_history)-1}"
                streamlit_feedback(
                    feedback_type="thumbs",
                    key=response_id,
                    align="flex-start",
                    on_submit=lambda feedback, rid=response_id: handle_feedback(feedback, rid)
                )
            except Exception as e:
                error_message = f"Error en el procesamiento: {str(e)}"
                st.write(error_message)
                st.session_state.chat_history.append(AIMessage(content=error_message))

if __name__ == "__main__":
    main()