"""
License:
- This code is released under the Apache License 2.0. See LICENSE file for full license text.

Copyright:
- Original Code (2024) by Balaji Lakshmanan.
- Code Revisions (2024) by Shad Nygren / Virtual Hipster Corporation.

Contributions:
- Added comprehensive docstrings and doctests to improve documentation and testability (Contributed by Shad Nygren, with assistance from ChatGPT).
- Enhanced robustness through comprehensive try-except error handling and refined code structure for better readability and maintenance (Contributed by Shad Nygren, with assistance from ChatGPT).

Note:
- Contributions by ChatGPT were under the guidance and specifications provided by the code's authors, ensuring alignment with project goals and standards.
"""

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


# Initialize logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Attempt to read the configuration file
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    if len(config.sections()) == 0:
        raise FileNotFoundError("The 'config.ini' file was not found or is empty.")
except FileNotFoundError as e:
    logging.error(f"Failed to read the configuration file: {e}")
    sys.exit("Exiting due to missing or empty configuration file.")

# Safely retrieve configuration values with error handling
def get_config_value(section, key, is_boolean=False):
    try:
        if is_boolean:
            return config.getboolean(section, key)
        else:
            return config[section][key]
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        logging.error(f"Missing section/key in configuration file: {e}")
        sys.exit(f"Exiting due to missing configuration for '{section}' -> '{key}'.")
    except ValueError as e:
        logging.error(f"Invalid value in configuration file: {e}")
        sys.exit(f"Exiting due to invalid value for '{section}' -> '{key}'.")

# Use the safe retrieval function to get config values
src_data_dir = get_config_value('index', 'src_data_dir')
basic_idx_dir = get_config_value('index', 'basic_idx_dir')
sent_win_idx_dir = get_config_value('index', 'sent_win_idx_dir')
auto_mrg_idx_dir = get_config_value('index', 'auto_mrg_idx_dir')
modelname = get_config_value('index', 'modelname')
embed_modelname = get_config_value('index', 'embedmodel')
useopenai = get_config_value('index', 'useopenai', is_boolean=True)

#--------------------

def check_and_create_directory(directory_path):
    """
    Checks if a directory exists at the specified path. If not, creates the directory.
    
    This function now includes error handling to catch exceptions that may occur during the 
    directory check or creation process, such as permissions issues or filesystem errors. 
    Error messages are logged, providing clear feedback for troubleshooting.
    
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
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Directory '{directory_path}' created successfully.")
        else:
            logging.info(f"Directory '{directory_path}' already exists.")
    except PermissionError:
        logging.error(f"Permission denied: Unable to access or create the directory '{directory_path}'. Check your permissions.")
        raise
    except OSError as e:
        logging.error(f"OS error occurred while accessing or creating the directory '{directory_path}'. Error: {e}")
        raise

#--------------------

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

#--------------------

def construct_sentencewindow_index(src_directory_path, index_directory):    
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
    try:
        check_and_create_directory(index_directory)
    except Exception as e:
        logging.error(f"Failed to create or access the index directory '{index_directory}'. Ensure it is a valid path and writable. Error: {e}")
        raise SystemExit("Exiting due to directory access issue.")

    try:
        if useopenai:
            llm = ChatOpenAI(temperature=0.1, model_name=config['api']['openai_modelname'])
        else:
            llm = LlamaCpp(
                model_path="./models/" + config['api']['local_modelname'],
                n_gpu_layers=40,  # Adjust based on your model and GPU VRAM
                n_batch=4096,
                n_ctx=4096,
                n_threads=8,
                temperature=0.1,
                f16_kv=True
            )
    except Exception as e:
        logging.error(f"Failed to initialize the language model for sentence window index construction. Check the model configuration and paths. Error: {e}")
        raise SystemExit("Exiting due to language model initialization issue.")

    try:
        documents = SimpleDirectoryReader(src_directory_path).load_data()
    except Exception as e:
        logging.error(f"Failed to read documents from '{src_directory_path}'. Check if the directory exists, is readable, and contains valid documents. Error: {e}")
        raise SystemExit("Exiting due to document reading issue.")

    try:
        index = build_sentence_window_index(
            documents,
            llm,
            embed_model=embed_modelname,
            save_dir=index_directory
        )
    except Exception as e:
        logging.error(f"Failed to construct or persist the sentence window index. Check the documents' format, the service context, and the index directory's writability. Error: {e}")
        raise SystemExit("Exiting due to sentence window index construction or persisting issue.")

    return index

#--------------------

def construct_automerge_index(src_directory_path, index_directory):
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
    try:
        check_and_create_directory(index_directory)
    except Exception as e:
        logging.error(f"Failed to create or access the index directory '{index_directory}'. Ensure it is a valid path and writable. Error: {e}")
        raise SystemExit("Exiting due to directory access issue.")

    try:
        if useopenai:
            llm = ChatOpenAI(temperature=0.1, model_name=config['api']['openai_modelname'])
        else:
            llm = LlamaCpp(
                model_path="./models/" + config['api']['local_modelname'],
                n_gpu_layers=-1,  # This signifies using all available GPU layers; adjust as necessary.
                n_batch=4096,
                n_ctx=4096,
                n_threads=8,
                temperature=0.1,
                f16_kv=True
            )
    except Exception as e:
        logging.error(f"Failed to initialize the language model for automerge index construction. Check the model configuration and paths. Error: {e}")
        raise SystemExit("Exiting due to language model initialization issue.")

    try:
        documents = SimpleDirectoryReader(src_directory_path).load_data()
    except Exception as e:
        logging.error(f"Failed to read documents from '{src_directory_path}'. Check if the directory exists, is readable, and contains valid documents. Error: {e}")
        raise SystemExit("Exiting due to document reading issue.")

    try:
        index = build_automerging_index(
            documents,
            llm,
            embed_model=embed_modelname,
            save_dir=index_directory
        )
    except Exception as e:
        logging.error(f"Failed to construct or persist the automerge index. Check the documents' format, the service context, and the index directory's writability. Error: {e}")
        raise SystemExit("Exiting due to automerge index construction or persisting issue.")

    return index

#--------------------

# Function calls for constructing indexes with error handling
try:
    # Create basic index
    index = construct_basic_index(src_data_dir, basic_idx_dir)
except Exception as e:
    logging.error(f"Failed to construct the basic index: {e}")
    # Optionally, exit or handle the error appropriately
    sys.exit(1)

try:
    # Create sentence window index
    sentindex = construct_sentencewindow_index(src_data_dir, sent_win_idx_dir)
except Exception as e:
    logging.error(f"Failed to construct the sentence window index: {e}")
    # Optionally, exit or handle the error appropriately
    sys.exit(1)

try:
    # Create automerge index
    autoindex = construct_automerge_index(src_data_dir, auto_mrg_idx_dir)
except Exception as e:
    logging.error(f"Failed to construct the automerge index: {e}")
    # Optionally, exit or handle the error appropriately
    sys.exit(1)
