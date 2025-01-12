CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Only include the generated Cypher statement in your response.

Always use case insensitive search when matching strings.

Schema:
{schema}

Examples: 
# Use case insensitive matching for entity ids
MATCH (c:Chunk)-[:HAS_ENTITY]->(e)
WHERE e.id =~ '(?i)entityName'

# Find documents that reference entities
MATCH (d:Document)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(e)
WHERE e.id =~ '(?i)entityName'
RETURN d

The question is:
{question}"""


cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    exclude_types=["Session", "Message", "LAST_MESSAGE", "NEXT"],
    allow_dangerous_requests=True
)

'''Enhanced schema
If the properties within your knowledge graph contain a relatively small range of values, you may benefit from using the enhanced_schema parameter.

When you set the enhanced_schema parameter, the system scans property values and provides examples to the LLM when generating Cypher queries.

This can lead to more accurate Cypher queries, at the cost of more complex prompts, and potentially slower generation times.
'''

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    enhanced_schema=True,
    allow_dangerous_requests=True
)

'''LLM Configuration
You can configure the GraphCypherQAChain to use different LLMs for Cypher and question/answer generation.

Using different LLMs can give give improved performance and/or better cost efficiency. Picking the right LLM for the right task can be a trade-off between speed and accuracy.'''

qa_llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model="gpt-3.5-turbo",
)

cypher_llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model="gpt-4",
    temperature=0
)
cypher_chain = GraphCypherQAChain.from_llm(
    qa_llm=qa_llm,
    cypher_llm=cypher_llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True
)