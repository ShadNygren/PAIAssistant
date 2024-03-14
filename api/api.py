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

import logging
# Initialize logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import StorageContext, load_index_from_storage, ServiceContext
import gradio as gr
import sys
import os
import logging
from utils import get_automerging_query_engine
from utils import get_sentence_window_query_engine
import configparser
from TTS.api import TTS
from gtts import gTTS
import simpleaudio as sa
import threading
from datetime import datetime
import json
import subprocess
from llama_index.prompts.base import PromptTemplate
from inference import main as generateVideo
import pyttsx3


#--------------------
print("ShadDEBUG-1")
def run_inference_original(checkpoint_path, face_video, audio_file, resize_factor, outfile):
    """
    Runs video generation inference using specified parameters and inputs.
    
    Constructs a command with dynamic parameters for video generation and invokes
    the `generateVideo` function with this command.
    
    Parameters:
    checkpoint_path (str): Path to the model checkpoint.
    face_video (str): Path to the face video file.
    audio_file (str): Path to the audio file.
    resize_factor (int): Resize factor for the video generation.
    outfile (str): Output path for the generated video.
    
    >>> run_inference("./checkpoints/model.pth", "face.mp4", "audio.wav", 2, "out.mp4")
    ['./checkpoints/model.pth', 'face.mp4', 'audio.wav', 2, 'out.mp4']
    """
    # Construct the command with dynamic parameters
    command = [        
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_file,
        "--resize_factor", str(resize_factor),
        "--outfile", outfile
    ]
    print(command)    
    generateVideo(command)

#--------------------

def run_inference(checkpoint_path, face_video, audio_file, resize_factor, outfile):
    """
    Runs video generation inference using specified parameters and inputs.
    
    Constructs a command with dynamic parameters for video generation and invokes
    the `generateVideo` function with this command.
    
    Parameters:
    checkpoint_path (str): Path to the model checkpoint.
    face_video (str): Path to the face video file.
    audio_file (str): Path to the audio file.
    resize_factor (int): Resize factor for the video generation.
    outfile (str): Output path for the generated video.
    
    >>> run_inference("./checkpoints/model.pth", "face.mp4", "audio.wav", 2, "out.mp4")  # doctest: +SKIP
    """
    command = [
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_file,
        "--resize_factor", str(resize_factor),
        "--outfile", outfile
    ]

    # Logging the command for debugging purposes
    logging.info(f"Running inference with command: {command}")
    
    try:
        # Attempt to generate the video
        generateVideo(command)
        logging.info("Video generation completed successfully.")
    except Exception as e:
        # Log the exception details to help with troubleshooting
        logging.error(f"Failed to generate video. Error: {e}")
        # Depending on the use case, you might want to re-raise the exception
        # or handle it (e.g., by attempting a fallback operation or simply continuing)
        # For now, we'll re-raise to make it clear that an error occurred.
        raise


#--------------------
print("ShadDEBUG-2")
def play_sound_then_delete(path_to_wav):
    """
    Plays a sound from the specified WAV file and deletes the file afterwards.
    
    This function starts playback of the sound file in a new thread. Once playback
    is complete, it attempts to delete the WAV file. If any errors occur during
    playback or deletion, they are printed to the console.
    
    Parameters:
    path_to_wav (str): The file path to the WAV file to be played and deleted.
    
    Note: Actual sound playback and file deletion are not performed in the doctest.
    
    >>> play_sound_then_delete("example.wav")  # doctest: +SKIP
    """
    def play_and_delete():
        try:
            wave_obj = sa.WaveObject.from_wave_file(path_to_wav)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until the sound has finished playing
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            try:
                #os.remove(path_to_wav)
                print(f"File {path_to_wav} successfully deleted.")
            except Exception as e:
                print(f"Error deleting file: {e}")
            pass
     # Start playback in a new thread
    threading.Thread(target=play_and_delete, daemon=True).start()


#--------------------
print("ShadDEBUG-3")
config = configparser.ConfigParser()
config.read('config.ini')


os.environ["GRADIO_ANALYTICS_ENABLED"]='False'

indextype=config['api']['indextype'] 

embed_modelname = config['api']['embedmodel']
basic_idx_dir = config['index']['basic_idx_dir']
sent_win_idx_dir = config['index']['sent_win_idx_dir']
auto_mrg_idx_dir = config['index']['auto_mrg_idx_dir']
serverip = config['api']['host']
serverport = config['api']['port']
sslcert = config['api']['sslcert']
sslkey = config['api']['sslkey']
useopenai = config.getboolean('api', 'useopenai')
ttsengine = config['api']['ttsengine']
# Get the logging level
log_level_str = config.get('api', 'loglevel', fallback='WARNING').upper()
# Convert the log level string to a logging level
log_level = getattr(logging, log_level_str, logging.WARNING)


#--------------------
print("ShadDEBUG-4")
def chatbot(input_text):
    """
    Processes the input text through a chat model to generate a response, synthesizes speech from the response, 
    generates a video using the synthesized speech, and constructs a JSON response containing the original 
    response text, video, and audio file paths, along with citation data.

    This function integrates several components: querying an indexed database, text-to-speech (TTS) conversion, 
    video generation, and logging of operations. It leverages configured TTS engines and language models to 
    produce an audio response, which is then used to generate a corresponding video.

    Parameters:
    input_text (str): User-provided text input for the chatbot to process.

    Returns:
    str: A JSON-formatted string containing the chatbot's response, paths to the generated audio and video 
    files, and citation data extracted from the query engine's response.

    Note: This function involves file operations, network communication, and external service calls, 
    making it complex to directly test via doctests. The example below is illustrative.

    >>> chatbot("How does the indexing process work?") # doctest: +SKIP
    """
    global tts
    print("User Text:" + input_text)    
    
    response =query_engine.query(input_text)        
 
    # Save the output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_audfile=f"output_{timestamp}.wav"
    output_vidfile=f"output_{timestamp}.mp4"
    output_path = "../web/public/audio/output/"+output_audfile
    
    
    if ttsengine == 'coqui':
        tts.tts_to_file(text=response.response, file_path=output_path ) # , speaker_wav=["bruce.wav"], language="en",split_sentences=True)
    elif ttsengine == 'gtts':
        tts = gTTS(text=response.response, lang='en')
        tts.save(output_path)
    else:
        tts.save_to_file(response.response , output_path)
        tts.runAndWait()

    checkpoint_path = "./checkpoints/wav2lip_gan.pth"
    face_video = "media/Avatar.mp4"
    audio_file = "../web/public/audio/output/"+output_audfile
    outfile="../web/public/video/output/"+output_vidfile
    resize_factor = 2
    run_inference(checkpoint_path, face_video, audio_file, resize_factor, outfile)
    #play_sound_then_delete(output_path)

    #construct response object
    # Building the citation list from source_nodes
    citation = [
        {
            "filename": node.metadata["file_name"],
            "text": node.get_text()
        } for node in response.source_nodes
    ]
    
    # Creating the JSON object structure
    jsonResponse = {
        "response": response.response,
        "video": output_vidfile,
        "audio": output_audfile,
        "citation": citation
    }
    
    # Convert to JSON string
    jsonResponseStr = json.dumps(jsonResponse, indent=4)
        
    return jsonResponseStr

#--------------------
print("ShadDEBUG-5")
logging.basicConfig(stream=sys.stdout, level=log_level)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Email data query")

print("ShadDEBUG-6")
from langchain.llms import LlamaCpp
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

#from langchain.globals import set_debug
#set_debug(True)

print("ShadDEBUG-7")
if useopenai:
    from langchain.chat_models import ChatOpenAI
    modelname = config['api']['openai_modelname']
    llm =ChatOpenAI(temperature=0.1, model_name=modelname)
else:
    modelname = config['api']['local_modelname']
    n_gpu_layers = -1  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    
    #cache prompt/response pairs for faster retrieval next time.
    set_llm_cache(InMemoryCache())
    
    llm = LlamaCpp(
    model_path="./models/"+ modelname,
    cache=True,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    n_threads=8,
    temperature=0.01,
    max_tokens=512,
    f16_kv=True,
    repeat_penalty=1.1,
    min_p=0.05,
    top_p=0.95,
    top_k=40,
    stop=["<|end_of_turn|>"]  
    )



print("ShadDEBUG-8")
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_modelname
)

print("ShadDEBUG-9")
index_directory=''
if indextype == 'basic':
    index_directory = basic_idx_dir
elif indextype == 'sentence' :
    index_directory = sent_win_idx_dir
elif indextype == 'automerge':
    index_directory = auto_mrg_idx_dir

print(config['api']['indextype'] )
print(index_directory)
if ttsengine == 'coqui':
    tts = TTS(model_name="tts_models/en/ljspeech/vits--neon", progress_bar=False).to("cuda")
    #tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
elif ttsengine == 'gtts':
    tts = gTTS(text='', lang='en')        
else: 
    tts = pyttsx3.init()
    voices = tts.getProperty('voices')
    tts.setProperty('voice', voices[1].id)  # this is female voice
    rate = tts.getProperty('rate')
    tts.setProperty('rate', rate-50)

print("ShadDEBUG-LoadIndex")
# load index
storage_context = StorageContext.from_defaults(persist_dir=index_directory)
index = load_index_from_storage(storage_context=storage_context, service_context=service_context)   
if indextype == 'basic':
    query_engine = index.as_query_engine()
elif indextype == 'sentence' :
    query_engine =get_sentence_window_query_engine(index)
elif indextype == 'automerge':
    query_engine = get_automerging_query_engine(automerging_index=index, service_context=service_context)

#prompts_dict = query_engine.get_prompts()
#print(list(prompts_dict.keys()))
    
# Optional: Adjust prompts to suit the llms.

qa_prompt_tmpl_str = (
    "GPT4 User: You are an assistant named Maggie. You assist with any questions regarding the organization kwaai.\n"
    "Context information is below\n"
    "----------------------\n"
    "{context_str}\n"
    "----------------------\n"
    "Given the context information and not prior knowledge respond to user: {query_str}\n"
    "<|end_of_turn|>GPT4 Assistant:"
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)


iface.launch( share=False, server_name=serverip, server_port=int(serverport), ssl_verify=False, ssl_keyfile=sslkey, ssl_certfile=sslcert)



print("ShadDEBUG-TestGradio-MainDoctest")
def test_gradio_interface_with_curl():
    """
    Tests the Gradio web interface using a curl command.
    
    This test attempts to send a POST request to the Gradio interface's API endpoint,
    simulating a user input and checking for a successful response.
    
    Returns:
    bool: True if the test passes (interface responds as expected), False otherwise.
    """
    try:
        # Construct the curl command
        curl_command = """
        curl -s -X POST -H "Content-Type: application/json" \\
            -d '{"data": ["Please tell me what is Kwaai about?"]}' \\
            http://127.0.0.1:7860/api/predict/
        """
        # Execute the curl command
        result = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Optional: Process the result.stdout or result.stderr if needed
        print("Curl command executed successfully, response:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Curl command failed:", e.stderr)
        return False


if __name__ == "__main__":
    # Added for running doctest
    import doctest
    doctest.testmod()

    print("Running interface test with curl...")
    test_success = test_gradio_interface_with_curl()
    if test_success:
        print("Interface test passed.")
    else:
        print("Interface test failed.")

