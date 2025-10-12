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
5. Presently I'm using LLMGraphTransformer from langchain_experimental.graph_transformers to build the graph using all the node types and all the relationship types but a better Top-Down approach would be to let the use to upload his/her own Ontology as the LLMs can very creative, having a domain specific Ontology will help improve the graph to a large extend. 
6. Use open source alternative to build and maintain graphs like Cognee to build graphs, Graphiti to keep graphs updated, etc.
7. I've been testing the system with only 1 document need to stress test with 100+ documents to see how the graph evolves with time. Need to improve the graph evolving and re-organizing strategy as well.
8. Ideally the passwords etc are retrieved at run-time from a password manager, but for testing keeping them hard-coded for now.


Hi, big fan!
If I don't win the Hackathon! Would love to meet the team in Bangalore for a coffee. ✌️