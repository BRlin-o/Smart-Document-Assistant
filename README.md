# Smart Document Assistant: A Case Study with Gogoro Vehicle Manual, AI Makes It Easy!

https://github.com/BRlin-o/Smart-Document-Assistant/assets/34859596/a62099cb-f917-4511-aceb-fe174ab2e858

🚀  This project, created for the #NVIDIADevContest using #LangChain and NVIDIA NIM APIs, is a smart document assistant that uses the Gogoro scooter manual as an example. It's designed to help you understand complex instructions easily! 

💡  Simply ask your question, and the Agent will interact with you in natural language. It uses the ReAct framework to transform your question into a precise query and find the answer from the Gogoro manual.  **What's even better is you can choose the specific Gogoro model and language for your query!**

🔍  The project converts the PDF to Markdown format and uses FAISS to build a vector database, making search faster and more accurate.

🖼️  To display images from the document within the conversation, technical challenges were overcome by using Markdown conversion, saving images separately, and having the front-end support Markdown content output, allowing images to be easily presented.

💬  For prompt design, three different prompts were created: ReAct prompts, which combine specific scenarios, vehicle types, and languages; tool prompts, specifically designed for different tool characteristics; and RAG manual-related prompts, improving the quality of answers to manual-related questions.

🌐 It is particularly emphasized that the document used in this project demo, the Gogoro manual, is in Chinese. Users can communicate with the Agent in English, but the Agent can retrieve and respond to queries from the Chinese manual content, enabling bilingual interaction. In the future, more language support can be implemented to accommodate a wider range of users.

🚀  The video showcases the AI answering various questions about Gogoro vehicles, such as:
- Where is the smart key card sensor located?
- What is the function of vitamin D? (Irrelevant questions will not be answered.)
- How to turn on and off the system power of Gogoro electric scooter?
- How to use mobile phone as key?
- How long is the regular maintenance interval for the car?

💻  With the powerful capabilities of LangChain and NVIDIA NIM APIs, you can easily create your own smart document assistant, turning any document into an intelligent helper!

#LangChain #NVIDIADevContest 
