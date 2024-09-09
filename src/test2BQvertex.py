from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import bigquery
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI # Cambiado a Vertex AI
import streamlit as st
import os

load_dotenv()

google_credentials_dir = os.getenv("GOOGLE_CLOUD_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_dir

def init_bigquery_client():
    client = bigquery.Client()
    return client

if "client" not in st.session_state:
    st.session_state.client = init_bigquery_client()

def get_bigquery_chain(client, dataset_id):
    template = """
      You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
      Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

      <SCHEMA>{schema}</SCHEMA>

      All table names must be fully qualified with the dataset name {dataset_id}. For example, if the table name is granosbasicos, it
      should be referenced as {dataset_id}.granosbasicos in the SQL Query.

      Remember the table granosbasicos stores information about corn, beans, sorghum, and information related to vegetables in 
      the table hortalizas. Important: all string values in the where clause must be capitalized.

      Conversation History: {chat_history}

      Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

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
    llm = ChatVertexAI(model="gemini-1.5-pro-001")  # Cambiado a Vertex AI

    def get_schema(_):
        # Leer documentacion de client.list_tables(dataset_id) y client.get_table(table_id) de 
        query = f"""
            SELECT 
            table_name, 
            column_name, 
            data_type FROM ai-mag-431021.CHATBOT.INFORMATION_SCHEMA.COLUMNS
            ORDER BY 
                table_name, 
                ordinal_position
    """
    
        results = client.query(query).result()
        
        schema_info = []
        for row in results:
            schema_info.append(f"Table: {row.table_name}, Column: {row.column_name}, Type: {row.data_type}")
        
        return "\n".join(schema_info)

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, client, chat_history: list, dataset_id: str):
    sql_chain = get_bigquery_chain(client, dataset_id)

    template =""" 
        You are a data analyst at a company. You are interacting with a users asking you questions about the company's database, you will not answer 
        any question outside of the scope of your data and will categorically reply that you won't. When answering questions about quantity of 
        production, add quintals, for example: 300 quintals of corn. And when they are about production area, answer with following word after 
        the quantity, for example: 100 manzanas. Use commas to separate thousands and millions and two decimals, for example: 1,100,200.00.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatVertexAI(model="gemini-1.5-pro-001")  # Cambiado a Vertex AI

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: client.list_tables(dataset_id),  # prueba metodo list_table, evaluar cambiar en la funcion schema
            #response=lambda vars: client.query(vars["query"].strip()).result().to_dataframe(),
            response=lambda vars: (
                print("Generated SQL Query:", vars["query"]),  # Imprimir la consulta SQL
                exec_query(client, vars["query"])  # Ejecutar la consulta
            )[1],
            #response=lambda vars: client.query(vars["query"].replace('sql', '').replace('', '').strip()).result().to_dataframe(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream({
        "question": user_query,
        "chat_history": chat_history,
        "dataset_id": dataset_id,
    })

def exec_query(client, query): #maybe to handle the error when execute the sql
    try:
        return client.query(query.replace('sql','').replace('','').strip()).result().to_dataframe()
    except Exception as e:
        return f"Error executing SQL: {str(e)}"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola soy Asistente OIAD, un chatbot para solventar dudas de siembra y producción de granos basicos y hortalizas."
                  "Recuerda que me encuentro en una etapa de desarrollo, la información que genero no es oficial y debe ser validada."),
    ]

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
        .st-emotion-cache-czk5ss.e16jpq800
        {
            visibility: hidden;
        }
        .stDeployButton
        {
            visibility: hidden;
        }
            
    </style>
""", unsafe_allow_html=True)# 

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

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
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
        response = get_response(user_query, st.session_state.client, st.session_state.chat_history, dataset_id)
        ai_response = st.write_stream(response)
    
    st.session_state.chat_history.append(AIMessage(content=ai_response))