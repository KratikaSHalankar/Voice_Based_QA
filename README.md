# Voice_Based_QA

A voice-activated system that answers questions about India using speech recognition and text-to-speech technologies.

    This project implements a voice-activated question answering system that answers questions about India based on predefined context. 
The system uses speech recognition to take user input, processes it with a machine learning model for question answering, and responds with a spoken answer using text-to-speech (TTS).


## Key Areas: 

**1.Context:** Your context about India, its states, and capitals is hardcoded into the program. It can be updated to include more dynamic content or sources. 

**2.Speech Processing:** Your program uses Google’s Speech Recognition API and the microphone for capturing audio, which might need proper setup depending on the environment. 

**3.Model:** The QA model deepset/roberta-base-squad2 is designed to provide answers based on the context. It is tuned for the SQuAD2.0 dataset, making it suitable for factual question answering.


## How It Works:

**1.Get Audio Input:** The program listens to the user's question through a microphone. 

**2.Convert Speech to Text:** The speech_recognition library converts the spoken question into text. 

**3.Generate an Answer:** The text is passed to a pre-trained question-answering model from Hugging Face (in this case, deepset/roberta-base-squad2), which retrieves the relevant answer from the provided context (about India).

**4.Convert Answer to Speech:** The answer is converted into speech using gTTS (Google Text-to-Speech) and then played through the speakers. 




