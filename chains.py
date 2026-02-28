from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

#reflection prompt and critique our twitter post
#going to suggest how it can be better and suggestions to improve it
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendatoins for the user's tweet"
            "Always provide detailed recommendatoins, including requests for length, virality, style, etc.",
        ),
        #place holder for history messages
        #agent will inoke and critique these
        MessagesPlaceholder(variable_name="messages")
    ]
)

#generation prompt
# generate the tweets that will be revised over and over again 
# after the feedback we get from the reflection prompt.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),

    ]
)


#initialize our LLM
llm = ChatOpenAI()

#create two chains 
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

