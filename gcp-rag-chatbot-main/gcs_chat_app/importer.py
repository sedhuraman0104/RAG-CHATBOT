# import all the file form wedding_list folder and emdbeding the file to database.
import os
import base64
import uuid
import json
from typing import (
    List,
    Dict,
    Any,
    Annotated,
    Union,
    Optional,
    Tuple,
    TypedDict,
    Literal,
)
import vertexai
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIImageCaptioning
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from tqdm import tqdm

DEBUG = False
RE_WRITE = True
IMG_FOLDER = os.getcwd() + "/img_output"
DATA_FOLDER = os.getcwd() + "/Training_01112025/"


class Parser:
    def __init__(self):
        env = load_dotenv()
        if os.environ.get("stage") == "dev":
            self.connection_string = (
                "postgresql+psycopg2://user:password@127.0.0.1:5432/vector-db"
            )
            self.prompt_llm = ChatOllama(model="llama3.1", temperature=0.1)
            self.json_llm = ChatOllama(model="llama3.1", temperature=0.1, format="json")
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
            )
        else:
            PROJECT_ID = "ctg-rag-model-001"
            LOCATION = "us-central1"
            vertexai.init(project=PROJECT_ID, location=LOCATION)

            self.connection_string = os.getenv("PROD_DB_URL")
            self.prompt_llm = ChatVertexAI(
                model="gemini-pro",
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                max_tokens=256,
                max_retries=3,
                verbose=True,
                streaming=True,
                stop=None,
            )
            self.embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    def run(self):
        conglomerate_info_list = []

        for root, dirs, files in os.walk(DATA_FOLDER):
            for file in files:
                if file.endswith(".pdf"):
                    try:
                        print(f"Parser with {os.path.join(root, file)}")
                        self.clean_output_path()
                        metadata = self.parser_path_to_text(os.path.join(root, file))

                        already_seen = {d["name"] for d in conglomerate_info_list}

                        if metadata.get("name") != metadata.get("group"):
                            if metadata.get("name") not in already_seen:
                                conglomerate_info_list.append(metadata)

                        try:
                            documents = PyPDFLoader(
                                file_path=os.path.join(root, file), extract_images=False
                            ).load()
                            # docs = PdfReader(os.path.join(root,file), ).pages
                            # for i, page in enumerate(docs):
                            #     texts = page.extract_text()
                            #     for image in page.images:
                            #         with open(output_path + "/" + image.name, "wb") as fp:
                            #             fp.write(image.data)
                        except Exception as e:
                            if "/Filter" in repr(e):
                                documents = PyPDFLoader(
                                    file_path=os.path.join(root, file)
                                ).load()
                            else:
                                print(
                                    f"Error in loading PDF {os.path.join(root, file)}"
                                )
                                print(e)
                                continue
                        # image_documents = parser_pdf_image(os.path.join(root,file))
                        # documents.extend(image_documents)

                        if RE_WRITE:
                            for doc in documents:
                                print(
                                    f"- Rewriting content with page: {doc.metadata['page']}"
                                )

                                prompt = f"""
                                    You are a content rewriter, please state only the truth and rewrite the 
                                    following content for better understanding only in English. 
                                    Don't add any new information which is not existed in the content.
                                    Remove all the contact information and any other personal information.
                                    Output the changed content only in the response with English only 
                                    without the more explanation or wording.  
                                    
                                    Content:
                                    {doc.page_content}
                                """.strip()
                                messages = [
                                    (
                                        "system",
                                        "You are a content rewriter, please state only the truth and rewrite "
                                        "the following content for better understanding.",
                                    ),
                                    ("human", prompt),
                                ]
                                # template = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
                                # chain = template | self.prompt_llm
                                updated_prompt = self.prompt_llm.invoke(messages)
                                if DEBUG:
                                    print("Original Prompt: ", doc.page_content)
                                    print("Updated Prompt: ", updated_prompt.content)

                                if "need more" in updated_prompt.content:
                                    continue
                                elif "don't know" in updated_prompt.content:
                                    continue
                                elif "no content provided" in updated_prompt.content:
                                    continue
                                else:
                                    doc.page_content = updated_prompt.content

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=600, chunk_overlap=80
                        )
                        texts = text_splitter.split_documents(documents)
                        for doc in texts:
                            doc.metadata.update(metadata)

                        db = PGVector.from_documents(
                            embedding=self.embeddings,
                            documents=texts,
                            connection_string=self.connection_string,
                            collection_name="default",
                        )

                        db = PGVector.from_documents(
                            embedding=self.embeddings,
                            documents=texts,
                            connection_string=self.connection_string,
                            collection_name=metadata.get("collection"),
                        )

                        print(f"Embedding Metadata: {metadata}")
                        print(f"Embedding is done with {os.path.join(root, file)}")
                    except Exception as e:
                        print(f"Error in embedding {os.path.join(root, file)}")
                        print(e)

        # embed all the conglomerate_info_list to database
        all_conglomerate_info = ""
        for conglomerate in conglomerate_info_list:
            conglomerate_string = (
                f"'{conglomerate.get('name')}' conglomerate which is belong to '{conglomerate.get('group')}' group and located in '{conglomerate.get('country')}' country"
                f", '{conglomerate.get('region')}' region and '{conglomerate.get('city')}' city, and the collection ID is '{conglomerate.get('collection')}'\n"
            )
            # conglomerate_meta = json.dumps(conglomerate)
            all_conglomerate_info += conglomerate_string

        documents = Document(
            page_content=all_conglomerate_info,
            metadata={
                "source": "conglomerate_info",
                "page": 0,
            },
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        texts = text_splitter.split_documents([documents])

        COLLECTION_NAME = "index"
        db = PGVector.from_documents(
            embedding=self.embeddings,
            documents=texts,
            connection_string=self.connection_string,
            collection_name=COLLECTION_NAME,
        )
        print(f"Embedding is done with the conglomerate list:\n {all_conglomerate_info}")

    def clean_output_path(self):
        for file in os.listdir(IMG_FOLDER):
            file_path = os.path.join(IMG_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def encode_image(self, image_path: str):
        with open(image_path, "rb") as f:
            return "data:image/jpg;base64," + base64.b64encode(f.read()).decode("utf-8")

    def parser_path_to_text(self, file_path: str):
        conglomerate_group = file_path.split(DATA_FOLDER)[1].split("/")[0].rstrip().lstrip()
        conglomerate_name = file_path.split(DATA_FOLDER)[1].split("/")[-2].rstrip().lstrip()
        conglomerate_dict = {}

        if "Information" in conglomerate_name:
            conglomerate_name = conglomerate_group
            template = {
                "name": conglomerate_name,
                "group": conglomerate_group,
                "country": "",
                "region": "",
                "city": "",
                "collection": conglomerate_group.lower().replace(" ", "_"),
                "description": "general information",
            }
            return template

        prompt = f"""
            Follow below example and introduction to identify all information in the below JSON table.
            Fill out the group, country, region, city, collection and description information in the JSON table.
            Only return the data in JSON format in the response without the more explanation or wording.

            Example:
            {{
                [
                    {
                    "name": "Solara Nexus",
                    "group": "Solara Conglomerate",
                    "country": "Fictivia",
                    "region": "Northern Corridor",
                    "city": "NeoBay",
                    "collection": "solara_global",
                    "description": "Solara Nexus is a multifunctional complex located in the heart of NeoBay, Fictivia. As part of the Solara Conglomerate's global collection, it serves as a hub for innovation, collaboration, and engagement. The facility offers modular spaces, advanced infrastructure, and adaptive environments designed to support a wide range of initiatives, including research, creative development, and immersive experiences. Its strategic location and flexible design make it a key node in the conglomerate’s international network."
                    }
                ]
            }}

            JSON Table:
            {{
              [
                {{
                  "name": "{conglomerate_name}",
                  "group": "{conglomerate_group}",
                  "country": "",
                  "region": ""
                  "city": "",
                  "collection": "",
                  "description": ""
                }},
              ]
            }}
            """

        messages = [
            (
                "system",
                "You are a helpful assistant. Extract information from the following JSON table.",
            ),
            ("human", prompt),
        ]

        response = self.json_llm.invoke(messages)
        response_content = response.content
        conglomerate_info = json.loads(response_content)
        return conglomerate_info

    def summarize_image(encoded_image):
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the contents of this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            ),
        ]
        response = VertexAIImageCaptioning().invoke(input=encoded_image)
        return response

    def extract_image(self, file_name: str):
        # Open the PDF file
        reader = PdfReader(file_name)
        for page in reader.pages:
            for image in page.images:
                with open(IMG_FOLDER + "/" + image.name, "wb") as fp:
                    fp.write(image.data)
        print(f"All images extracted to {IMG_FOLDER}")

    def pdf_loader(self, file_name: str):
        text_elements = []
        table_elements = []

        text_summaries = []
        table_summaries = []

        summary_prompt = """
        Summarize the following {element_type}:
        {element}
        """
        summary_chain = LLMChain(
            llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024),
            prompt=PromptTemplate.from_template(summary_prompt),
        )

        raw_pdf_elements = partition_pdf(
            filename=file_name,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=IMG_FOLDER,
        )
        raw_pdf_elements = partition_pdf(
            filename=file_name,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=IMG_FOLDER,
        )
        for elem in raw_pdf_elements:
            if "CompositeElement" in repr(elem):
                text_elements.append(elem.text)
                summary = summary_chain.run(
                    {"element_type": "text", "element": elem.text}
                )
                text_summaries.append(summary)

            elif "Table" in repr(elem):
                table_elements.append(elem.text)
                summary = summary_chain.run(
                    {"element_type": "table", "element": elem.text}
                )
                table_summaries.append(summary)

    def parser_pdf_image(self, file_name: str):
        try:
            self.extract_image(file_name)
            # Get image summaries
            image_elements = []
            image_summaries = []
            documents = []
            retrieve_contents = []

            for img in sorted(
                os.listdir(IMG_FOLDER),
                key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
            ):
                if img.endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(IMG_FOLDER, img)
                    encoded_image = self.encode_image(image_path)
                    image_elements.append(encoded_image)
                    summary = self.summarize_image(encoded_image)
                    image_summaries.append(summary)
                    print(f"Image summary with {image_path}, caption: {summary}")

            for elem, summary in zip(image_elements, image_summaries):
                uuid_index = str(uuid.uuid4())
                doc = Document(
                    page_content=summary,
                    metadata={
                        "id": uuid_index,
                        "type": "image",
                        "original_content": elem,
                        "source": file_name,
                        "page": 0,
                    },
                )
                retrieve_contents.append((uuid_index, summary))
                documents.append(doc)
            return documents

        except Exception as e:
            print(f"Error in parser_pdf_image {file_name}")
            print(e)
            return []


if __name__ == "__main__":
    parser = Parser()
    parser.run()
