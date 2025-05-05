# Sprint Planning

## Week 01

During the first week you will get a general overview of the main concepts and tools you will need to kickstart the Sprint project from day 1.

### Lesson 01

- Theory: 20.1 - Basics of Large Language Models (LLMs)
- Practice:
- Sprint project:
    - Get all the requirements installed in a virtual env and the chainlit app running.
    - Complete function extract_text_from_pdf() at backend/utils.py.

### Lesson 02

- Theory: 20.2 - Prompt Engineering
- Practice:
- Sprint project:
    - Complete the code for the class ChatAssistant() at backend/models/chatgpt_clone.py (We should modify the code for app.py so we can let the user load different assistant models).

## Week 02

In the second week we will introduce only a new concept, `Information Retrieval`, and then focus on the Sprint project.

### Lesson 01

- Theory: 20.3 - Information Retrieval, Embeddings, Chunking
- Practice:
- Sprint project:
    - Complete the code for the ETLProcessor class at backend/etl.py.
    - Run ingestion/etl.py to create the initial dataset with vector embeddings.

### Lesson 02

- Theory: 20.3 - Information Retrieval, Embeddings, Chunking
- Practice:
- Sprint project:
    - Complete the code for the JobsFinderAssistant() class at backend/models/jobs_finder.py. This class should consume the jobs from the database we've created in the previous lesson.

## Week 03

In the third week we will introduce the last concept, `Tools, Agents and Chat Memory`, and then focus on the Sprint project.

### Lesson 01

- Theory: 20.4 - Tools, Agents and Chat Memory
- Practice:
- Sprint project:
    - Complete the function build_cover_letter_writing() at backend/models/jobs_finder_agent.py. This class should consume the jobs from the database we've created in the previous lesson but also have some extra tools.
    - The code for this one is not complete, we as instructors should think what makes sense for an agent to do and what not. Maybe we can reuse the send email functionality Claudio has built but we need to explicity teach students how to configure that and may be hard.

### Lesson 02

- Theory: -
- Practice:
- Sprint project:
    - Take the profile + job found and make a personalized message to the recruiter.
