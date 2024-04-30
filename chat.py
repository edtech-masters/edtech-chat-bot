import os
import chainlit as cl
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import AWSV4SignerAuth
import boto3
from opensearchpy import RequestsHttpConnection
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# set the environment variables
load_dotenv()

# load the environment variables AWS_OPENSEARCH_DOMAIN_ENDPOINT, AWS_PROFILE, AWS_REGION from the .env file otherwise exit the program
if os.getenv("AWS_OPENSEARCH_DOMAIN_ENDPOINT") is None and os.getenv("AWS_PROFILE") is None and os.getenv("AWS_REGION") is None:
    print("Please set the environment variables. Program will exit now.")
    exit()

if not os.getenv("AWS_PROFILE"):
    os.environ["AWS_PROFILE"] = 'default'

@cl.on_chat_start
async def on_chat_start():
    aws_opensearch_url = os.getenv("AWS_OPENSEARCH_DOMAIN_ENDPOINT")
    credentials = boto3.Session().get_credentials()
    region = os.getenv("AWS_REGION")
    awsauth = AWSV4SignerAuth(credentials, region)

    bedrock_embeddings = BedrockEmbeddings(credentials_profile_name=os.environ["AWS_PROFILE"], region_name=os.getenv("AWS_REGION"))
    vectorstore = OpenSearchVectorSearch(opensearch_url=aws_opensearch_url, 
        index_name='edtechmasters-content-index', 
        embedding_function=bedrock_embeddings,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatBedrock(
        model_id="anthropic.claude-v2",
        model_kwargs={"temperature": 0.1},
    )

    chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=create_stuff_documents_chain(llm, prompt=chat_prompt)
    )
    cl.user_session.set("chain", chain)
    
@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    response = chain.invoke({"input": message.content})
    await cl.Message(HumanMessage(content=response['answer']).content).send()

    

