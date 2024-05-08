from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Initialize QdrantClient
client = QdrantClient(host="localhost", port=6333)

embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the query vector and ensure it has the correct dimensions
contexts = """Q1. What are the top 5 Issue  with key of Facility Code, equipment name,equipment code and Date of WO, occurrences for PCR Curing Facility?  dont ansume use accurate from search result.
Q2. What are the top 5 Issue  occurrences for PCR Curing Facility based on Lost Hours list all?
Q3. Which Equipment Name for PCR Curing Facility has the Issue  of top1?
Q4. Which Equipment Name for PCR Curing Facility has the Issue  of top2?
Q5. Which Equipment Name for PCR Curing Facility has the maximum number of Issue  mentioned?
Q6. Is there any correlation between Shift in the A,B,C and Issue  data for PCR Curing Facility?"""

query_vector = embedding_function.embed_query(contexts)

# Perform the search operation
try:
    search_result = client.search(
        collection_name="public",
        query_vector=query_vector,
        with_vectors=False,
        with_payload=True,
        limit=100,
        offset=100,
    )

    # Print the search result
    print("Search result:")
    print(search_result)

except Exception as e:
    print("An error occurred during the search operation:")
    print(e)

Data_Engineer = Agent(
    role='data Engineer',
    goal='you are goal to identify, collect all points from search result payload its in json format as key use required tools from your knowledge and query_vector collect the al required data handover to agent data_scientist',
    backstory='Expert in analyzing and generating and collecting payload  from search result score from any format',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    search_result=search_result
)
# Creating Agents
Data_Scientist = Agent(
    role='data scientist',
    goal='you are goal to re-check the answer handovered by Data_Engineer,if required do the correction once its ready with accurate data handover to Data_Analyst',
    backstory='Expert in Re-checking the data, correction the answers ',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    search_result=search_result
)

Data_Analyst = Agent(
    role='Data Analyst',
    goal='your goal is to  analysis the answers handovered by Data_Scientist, show to accoutate answer to users',
    backstory='Expert in analysis, correction the answers',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    search_result=search_result
)

# Update the agent's response based on similarity
#Data_Scientist.update_response(search_result)
#Data_Analyst.update_response(search_result)

# Creating Tasks
data_identification_task = Task(
    description=f'Identify and summarize data relevant to the {contexts} from the search results.',
    agent=Data_Scientist,
    expected_output='Summarize data relevant to the provided {contexts} based on the search results.'
)

data_validation_task = Task(
    description=f'Analyze the corrected data and present the findings, Don\'t skip any topic.',
    agent=Data_Analyst,
    expected_output='Provide corrected summaries and ensure data accuracy.'
)

data_analysis_task = Task(
    description=f'Analyze the corrected data and present the findings, Don\'t skip any topic.',
    agent=Data_Engineer,
    expected_output='Provide a detailed analysis of the corrected data relevant to the {contexts}.if you don\'t skip any topic.'
)

# Creating Crew
answer_crew = Crew(
    agents=[Data_Scientist, Data_Analyst,Data_Engineer],
    tasks=[data_identification_task,data_validation_task, data_analysis_task],
    process=Process.sequential,
    manager_llm=llm,
    memory=True,
    search_result=search_result
)

# Execute the crew to see the result
result = answer_crew.kickoff()
print(result)
