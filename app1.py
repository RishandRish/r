
from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

client = QdrantClient(
    url="",
    api_key="",
)

embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the query vector and ensure it has the correct dimensions
contexts = """Q1. What are the top 5 Issue  with key of Facility Code  and equipment name,equipment code  from search result payload occurrences for PCR Curing Facility?  dont ansume use accurate from search result.
Q2. What are the top 5 Issue Summary occurrences for PCR Curing Facility based on Lost Hours?
Q3. Which Equipment Name for PCR Curing Facility has the Issue Summary of top1?
Q4. Which Equipment Name for PCR Curing Facility has the Issue Summary of top2?
Q5. Which Equipment Name for PCR Curing Facility has the maximum number of Issue Summary mentioned?
Q6. Is there any correlation between Shift and Issue Summary data for PCR Curing Facility?"""

query_vector = [0.0]* 1536

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
    goal='you are goal to identify, collect all points from search result score its in json formate as key use required tools from your knowlegde  and query_vector give to data_scientist',
    backstory='Expert in analyzing and generating and collecting scored points  from search result score handover to data_scientist.',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    search_result=query_vector
)
# Creating Agents
Data_Scientist = Agent(
    role='data scientist',
    goal='you are goal to Identify, research and identify from the serch result score',
    backstory='Expert in analyzing and generating key points from given data.',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    search_result=query_vector
)

Data_Analyst = Agent(
    role='Data Analyst',
    goal='your goal is to analyze the given data asked by the user and provide the best result.',
    backstory='Expert in crafting engaging narratives from complex information.',
    allow_delegation=True,
    verbose=True,
    llm=llm,
    search_result=query_vector
)

# Update the agent's response based on similarity
#Data_Scientist.update_response(search_result)
#Data_Analyst.update_response(search_result)

# Creating Tasks
data_identification_task = Task(
    description=f'use the same names as shown in search_result Identify the data of {contexts} from the search results score. Write the step-by-step summary for each question present in the context .',
    agent=Data_Scientist,
    expected_output='use the same names as shown in search_result score Give me the best step-by-step summary for each question present in the context based on the search results .'
)

data_analysis_task = Task(
    description=f'use the same names as shown in search_result score Go step by step.\nStep 1: Identify all the {contexts} received using the search results.\nStep 2: Go through every context and write an in-depth summary of the information retrieved . Don\'t skip any topic.',
    agent=Data_Analyst,
    expected_output='use the same names as shown in search_result score Provide the answers for each question present in the context based on the search results without assuming , if you don\'t know.'
)

user_question_taks = Task(
    description=f'use the same names as shown in search_result  score Go step by step.\nStep 1: Identify all the {contexts} received using the search results.\nStep 2: Go through every context and write an in-depth summary of the information retrieved . Don\'t skip any topic.',
    agent=Data_Analyst,
    expected_output='use the same names as shown in search_result  score Provide the answers for each question present in the context based on the search results without assuming , if you don\'t know.'
)

# Creating Crew
answer_crew = Crew(
    agents=[Data_Scientist, Data_Analyst],
    tasks=[data_identification_task, data_analysis_task],
    process=Process.sequential,
    manager_llm=llm,
    memory=True,
    search_result=search_result
)

# Execute the crew to see the result
result = answer_crew.kickoff()
print(result)
