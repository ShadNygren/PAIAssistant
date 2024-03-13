from llama_index import SimpleDirectoryReader 
from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex
from utils import build_sentence_window_index
from utils import build_automerging_index
from langchain.llms import LlamaCpp

import sys
import os
import logging
import configparser

config = configparser.ConfigParser()
config.read('config.ini')



# get config values
src_data_dir = config['index']['src_data_dir']
basic_idx_dir = config['index']['basic_idx_dir']
sent_win_idx_dir = config['index']['sent_win_idx_dir']
auto_mrg_idx_dir = config['index']['auto_mrg_idx_dir']
modelname = config['index']['modelname']
embed_modelname = config['index']['embedmodel']
useopenai = config.getboolean('index', 'useopenai')

        
def check_and_create_directory(directory_path):
    """
    Checks if a directory exists at the specified path. If not, creates the directory.
    
    Parameters:
    directory_path (str): The file system path where the directory should be checked/created.
    
    Returns:
    None
    
    >>> import tempfile, os, shutil
    >>> temp_dir = tempfile.mkdtemp()
    >>> test_dir = os.path.join(temp_dir, 'new_dir')
    >>> check_and_create_directory(test_dir)  # doctest: +ELLIPSIS
    Directory '...' created successfully.
    >>> os.path.isdir(test_dir)
    True
    >>> shutil.rmtree(temp_dir)  # Clean up the created temporary directory
    """
        
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
def construct_basic_index(src_directory_path,index_directory):
    """
    Constructs a basic index from documents in the specified source directory and saves it to the index directory.
    
    This function initializes a language model based on the configuration (OpenAI or a local model),
    reads documents from the source directory, creates a vector store index from these documents,
    and persists the index in the specified directory.

    Parameters:
    src_directory_path (str): The file path to the source directory containing documents to index.
    index_directory (str): The file path to the directory where the index should be saved.
    
    Returns:
    VectorStoreIndex: The constructed and persisted vector store index.

    >>> construct_basic_index('path/to/source', 'path/to/index') # doctest: +SKIP
    """
    try:
        check_and_create_directory(index_directory)
    except Exception as e:
        logging.error(f"Failed to create or access the index directory '{index_directory}'. Please ensure it is a valid path and writable. Error: {e}")
        raise SystemExit("Exiting due to directory access issue.")

    try:
        if useopenai:
            llm = ChatOpenAI(temperature=0.1, model_name=config['api']['openai_modelname'])
        else:
            llm = LlamaCpp(
                model_path="./models/"+config['api']['local_modelname'],
                n_gpu_layers=40,  # Adjust based on your model and GPU VRAM
                n_batch=4096,
                n_ctx=4096,
                n_threads=8,
                temperature=0.1,
                f16_kv=True
            )
    except Exception as e:
        logging.error(f"Failed to initialize the language model. Please check the model configuration and paths. Error: {e}")
        raise SystemExit("Exiting due to language model initialization issue.")

    try:
        documents = SimpleDirectoryReader(src_directory_path).load_data()
    except Exception as e:
        logging.error(f"Failed to read documents from '{src_directory_path}'. Please check if the directory exists, is readable, and contains valid documents. Error: {e}")
        raise SystemExit("Exiting due to document reading issue.")

    try:
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_modelname)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir=index_directory)
    except Exception as e:
        logging.error(f"Failed to construct or persist the index. Please check the documents' format, the service context, and the index directory's writability. Error: {e}")
        raise SystemExit("Exiting due to index construction or persisting issue.")

    return index

def construct_basic_index_original(src_directory_path,index_directory):
    check_and_create_directory(index_directory)     
    if useopenai:
        from langchain.chat_models import ChatOpenAI
        modelname = config['api']['openai_modelname']
        llm =ChatOpenAI(temperature=0.1, model_name=modelname)
    else:
        modelname = config['api']['local_modelname']
        n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        llm = LlamaCpp(
        model_path="./models/"+ modelname,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        n_threads=8,
        temperature=0.1,
        f16_kv=True
        )

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_modelname
    )
   
    documents = SimpleDirectoryReader(src_directory_path).load_data()
    index = VectorStoreIndex.from_documents(documents,
                                            service_context=service_context)
      
    index.storage_context.persist(persist_dir=index_directory)     
    return index

def construct_sentencewindow_index(src_directory_path,index_directory):    
    """
    Constructs a sentence window index from documents in the specified source directory. 
    This index is specifically designed to enhance search capabilities by considering the 
    context of sentences surrounding search terms. The constructed index is saved to the 
    specified index directory.

    Depending on the configuration, this function initializes a language model (OpenAI or 
    a local LlamaCpp model), reads documents from the source directory, and uses these 
    documents to build a sentence window index. This index is then persisted in the 
    provided directory.

    Parameters:
    src_directory_path (str): The file path to the source directory containing documents to index.
    index_directory (str): The file path to the directory where the index should be saved.
    
    Returns:
    The constructed sentence window index object. The specific type of this object depends 
    on the implementation of `build_sentence_window_index`.

    Example:
    Given the complexity and external dependencies involved in index construction, this 
    example is illustrative and not meant to be executed directly.

    >>> construct_sentencewindow_index('path/to/source', 'path/to/index') # doctest: +SKIP
    """

    if useopenai:
        from langchain.chat_models import ChatOpenAI
        modelname = config['api']['openai_modelname']
        llm =ChatOpenAI(temperature=0.1, model_name=modelname)
    else:
        modelname = config['api']['local_modelname']
        n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        llm = LlamaCpp(
        model_path="./models/"+ modelname,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        n_threads=8,
        temperature=0.1,
        f16_kv=True
        )
    documents = SimpleDirectoryReader(src_directory_path).load_data()
    index = build_sentence_window_index(
    documents,
    llm,
    embed_model=embed_modelname,
    save_dir=index_directory
    )
    return index

def construct_automerge_index(src_directory_path,index_directory):
    """
    Constructs an automerge index from documents in the specified source directory. 
    This index type supports merging similar documents automatically, enhancing the 
    search experience by consolidating information. The constructed index is saved to the 
    specified index directory.

    Depending on the configuration, this function initializes a language model (either 
    OpenAI or a local LlamaCpp model), reads documents from the source directory, and 
    employs these documents to build an automerge index. The index is subsequently 
    persisted in the designated directory.

    Parameters:
    src_directory_path (str): The file path to the source directory containing documents for indexing.
    index_directory (str): The file path to the directory where the index should be saved.
    
    Returns:
    The constructed automerge index object. The specific type of this object depends on the 
    implementation details of `build_automerging_index`.

    Example:
    Due to the function's complexity and its reliance on external dependencies, the following example 
    is provided for illustrative purposes and is not intended to be executed as part of doctests.

    >>> construct_automerge_index('path/to/source', 'path/to/index') # doctest: +SKIP
    """

    if useopenai:
        from langchain.chat_models import ChatOpenAI
        modelname = config['api']['openai_modelname']
        llm =ChatOpenAI(temperature=0.1, model_name=modelname)
    else:
        modelname = config['api']['local_modelname']
        n_gpu_layers = -1  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        llm = LlamaCpp(
        model_path="./models/"+ modelname,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        n_threads=8,
        temperature=0.1,
        f16_kv=True
        )
    documents = SimpleDirectoryReader(src_directory_path).load_data()
    index = build_automerging_index(
    documents,
    llm,
    embed_model=embed_modelname,
    save_dir=index_directory
    )
    return index
 
    
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
 
 
#Create basic index
index = construct_basic_index(src_data_dir,basic_idx_dir)
#create sentencewindow index
sentindex = construct_sentencewindow_index(src_data_dir,sent_win_idx_dir)
#create automerge index
autoindex = construct_automerge_index(src_data_dir,auto_mrg_idx_dir)
