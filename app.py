from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew,Process
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

# Initialize QdrantClient
client = QdrantClient(host="localhost", port=6333)

# Define the query vector and ensure it has the correct dimensions
query_vector = [0.0] * 1536  # Assuming 1536 dimensions for the query vector

embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")
# Perform the search operation
try:
    search_result = client.search(
        collection_name="public",
        query_vector=query_vector,
        with_vectors=True,
        with_payload=True,
        limit=10,
        offset=100,
    )

    # Print the search result
    print("Search result:")
    print(search_result)

except Exception as e:
    print("An error occurred during the search operation:")
    print(e)

# 2. Creating Agents
Data_Scientist = Agent(
    role='data scientist',
    goal='you are goal to Identify reasarch and identify the machine data ',
    backstory='Expert in analysing and generating key points from given data.',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    query_vector= query_vector
)

Data_Analyst = Agent(
    role='Data Analyst',
    goal='your goal is to analysis the given data ask user asked and give to best result.',
    backstory='Expert in crafting engaging narratives from complex information.',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    query_vector = query_vector
)

#topic ="{PCR Curing Facility}"

contexts ="""Q1. What are top 5 Issue Summary occurrences for PCR Curing Facility?
Q2. What are top 5 Issue Summary occurrences for PCR Curing Facility based on Lost Hours?
Q3. Which Equipment Name for PCR Curing Facility has Issue Summary of top1?
Q4. Which Equipment Name for PCR Curing Facility has Issue Summary of top2?
Q5. Which Equipment Name for PCR Curing Facility has maximum number of Issue Summary mentioned?
Q6. Is there any correlation between Shift and Issue Summary data for PCR Curing Facility?
"""
# 3. Creating Tasks
data_identification_task = Task(
    description='Identify the data from {query_vector}from {contexts} write the best answer without assuming by you',
    agent=Data_Scientist,
    expected_output='give me best result  from the {query_vector}from {contexts} without assuming ',
    query_vector=query_vector
)

data_analysis_task = Task(
    description="""
    Go step by step.
    Step 1: Identify all the {contexts} received using {query_vector}. 
    Step 2: Go through every contexts and write an in-depth summary of the information retrieved.
    Don't skip any topic.
    """,
    agent=Data_Analyst,
    expected_output='provide the all answers from the {query_vector}from {contexts} without assuming',
    query_vector=query_vector
)

# 4. Creating Crew
news_crew = Crew(
    agents=[Data_Scientist, Data_Analyst],  # Pass Agent instances directly
    tasks=[data_identification_task, data_analysis_task],  # Pass Task instances directly
    process=Process.sequential, 
    manager_llm=llm,
    query_vector=query_vector
)


# Execute the crew to see RAG in action
result = news_crew.kickoff()
print(result)
