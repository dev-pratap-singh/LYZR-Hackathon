The setup could be improved upon further investigations like
1. Using an SLM which can reduce the cost of creating the graph, study shows that up to 3 traversal of the documents results in the best graph which would be very coustly if done using OpenAI embedding models. I think using Gemma-3-8B 8-bit quantised model which take about 4 GBs can be loaded in memory in GGUF format and used to create the graphs on the go. 
2. Presently I'm using LLMGraphTransformer from langchain_experimental.graph_transformers to create graphs, the process of creating the graphs could be improved by uisng GraphRAG from Microsoft, which is a very well respected library, it 
    1. Perform a hierarchical clustering of the graph using the Leiden technique.
    2. Generate summaries of each community and its constituents from the bottom-up. This aids in holistic understanding of the dataset.
    3. At query time, these structures are used to provide materials for the LLM context window when answering a question. The primary query modes are:
        1. Global Search for reasoning about holistic questions about the corpus by leveraging the community summaries.
        2. Local Search for reasoning about specific entities by fanning-out to their neighbors and associated concepts.
        3. DRIFT Search for reasoning about specific entities by fanning-out to their neighbors and associated concepts, but with the added context of community information.
        4. Basic Search for those times when your query is best answered by baseline RAG (standard top k vector search).
3. Sometimes the context is lost when converting from an Image to text, which all these earlier techniques uses including vector search, graph search, filter search. Where the context lies spcially in the image, we can add a visual Image only Rag tool also to our agent to help with image queries:
Architecture:
PDF-2-Images -> Use a Late Interaction Model (Ex: ) + Multi Vector Embeddings -> Store it in a image-vector-embeddings friendly DB like Qdrant Vector DB -> Do retrieval (ViDoRe retrievel + Maxsum similarity btw the question embeddings & the image embeddings) -> Send the pages to Multi-Model LLM to answer the questions.
4. For a big document, the graph creation takes a lot of time, we could speed it up by parallel processing. We can use asyncio with semaphore to process them parallely.
