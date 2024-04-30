# This is just an example to show how to use Amazon OpenSearch Service, you need to set proper values.
import boto3
import boto3.session
from opensearchpy import RequestsHttpConnection
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
import os
from opensearchpy import AWSV4SignerAuth
from dotenv import load_dotenv
import sqlite3
import datetime

# set the environment variables
load_dotenv()

# load the environment variables AWS_OPENSEARCH_DOMAIN_ENDPOINT, AWS_PROFILE, AWS_REGION from the .env file otherwise exit the program
if os.getenv("AWS_OPENSEARCH_DOMAIN_ENDPOINT") is None and os.getenv("AWS_PROFILE") is None and os.getenv("AWS_REGION") is None:
    print("Please set the environment variables. Program will exit now.")
    exit()

if not os.getenv("AWS_PROFILE"):
    os.environ["AWS_PROFILE"] = 'default'

aws_opensearch_url = os.getenv("AWS_OPENSEARCH_DOMAIN_ENDPOINT")
credentials = boto3.Session().get_credentials()
region = os.getenv("AWS_REGION")
awsauth = AWSV4SignerAuth(credentials, region)

# list s3 bucket files
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket=os.getenv("AWS_BUCKET_NAME"))
files = response['Contents']

# connect to the sqlite database
db_connection = sqlite3.connect(f"{os.getenv("DB_NAME")}.db")
# create table 'processed_files' if it does not exist
db_connection.execute('CREATE TABLE IF NOT EXISTS processed_files (file_name TEXT PRIMARY KEY, processed BOOLEAN)')
db_connection.commit()

# clear the processed_files table
db_connection.execute('DELETE FROM processed_files')
db_connection.commit()
#exit()

# get the list of files that have not been processed
cursor = db_connection.cursor()
cursor.execute('SELECT file_name FROM processed_files WHERE processed = 1')
processed_files = cursor.fetchall()
processed_files = [file[0] for file in processed_files]
# process the files
for file in files:
    if file['Key'] not in processed_files:
        original_file_name = file['Key']
        
        # process the file
        # get the file from s3
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = file['Key'].split('/')[-1]
        new_file_name = f"{timestamp}_{file_name}"
        s3.download_file(os.getenv("AWS_BUCKET_NAME"), file['Key'], new_file_name)
        print(f"*** Processing {original_file_name} ***")

        # fileLoader define as null
        fileLoader = None
        # check file type and process accordingly
        if file_name.endswith('.txt'):
            fileLoader = TextLoader(new_file_name)
        elif file_name.endswith('.pdf'):
            fileLoader = PyPDFLoader(new_file_name)

        if fileLoader is not None:
            documents = fileLoader.load()
            # load the documents
            documents = fileLoader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            print(f"*** {len(docs)} documents created for {original_file_name} ***")
            
            # split the documents array into chunks of 500
            bulk_size = 500
            bulk_docs = [docs[i:i + bulk_size] for i in range(0, len(docs), bulk_size)]

            for docs_chunk in bulk_docs:    
                # get the embeddings
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region)
                embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
                vectorstore = OpenSearchVectorSearch.from_documents(
                    docs_chunk,
                    embeddings,
                    opensearch_url=aws_opensearch_url,
                    http_auth=awsauth,
                    timeout=300,
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection,
                    index_name="edtechmasters-content-index"
                )
                print(f"*** Vectorstore created for {original_file_name} ***")

        # insert file_name into processed_files
        db_connection.execute('INSERT INTO processed_files (file_name, processed) VALUES (?, ?)', (original_file_name, 1))
        db_connection.commit()
        # remove the file
        os.remove(new_file_name)

# close the cursor
cursor.close()

# close the connection
db_connection.close()
