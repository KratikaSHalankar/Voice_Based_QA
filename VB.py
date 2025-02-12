import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import speech_recognition as sr #to take the question from user as speech and will convert it to text
from transformers import pipeline #the questio that is the form of text will be procssed and pipeline has QA model that will generate answer to the question based on the context provided in the form of text.
from gtts import gTTS # converts the answer generated as text to speech
from playsound import playsound #function is used to play the audio file that was created by the gTTS library.
#it will get the question from user
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def get_audio_input():
    recognizer=sr.Recognizer()# recognizer is object that will store speech
    with sr.Microphone() as source: #will open the microphone to record the speech
        print("ask your question")
        recognizer.adjust_for_ambient_noise(source,duration=0.2) #it will adjust and focus on the speech
        audio =recognizer.listen(source)# the speech is captured and will be stored in the audio variable
    try:
        question=recognizer.recognize_google(audio)# convert the speech to text by recognizer.recongnize_google(audio)
        print(f"question:{question}")
        return question
    #if the speech is not clearly audible the it will jump to except
    except sr.UnknownValueError:
        print("cannot listen properly")
        return None
    except sr.RequestError as e:
        print(f"cannot generate the response:{e}")

# function that will generate answer based on the question asked by the user       
def generate_answer(question,context):
    qa_pipeline=pipeline("question-answering",model="deepset/roberta-base-squad2")
    result=qa_pipeline(context=context,question=question)#will call the function with parameter as context and question
    return result['answer']

#function that give the answer as speech 
def text_to_speech(text):
    tts=gTTS(text=text,lang='en')
    tts.save("answer.mp3")
    playsound("answer.mp3")

#main fuction to perform the all the operation
def main():
    context="""Geography and Country Information of India
India, officially known as the Republic of India, is located in South Asia. The capital of India is New Delhi. The country has two official languages: Hindi and English, though several regional languages are spoken throughout the nation. India is the 7th largest country in the world, covering an area of approximately 3.29 million square kilometers. It has a population of over 1.4 billion people. The currency used is the Indian Rupee (INR).
India shares borders with Pakistan, China, Nepal, Bhutan, Bangladesh, Myanmar, and Afghanistan. The country has a coastline of about 7,500 kilometers, with notable coastal regions like Goa and Kerala. The geography of India is diverse, ranging from the towering Himalayas in the north, the fertile Ganges plains, the vast Thar Desert in the west, to the lush tropical forests in the south. Major rivers like the Ganges, Brahmaputra, and Yamuna flow through the country, providing vital water sources.
India’s time zone is Indian Standard Time (IST), which is UTC +5:30.
States of India with Capitals, Languages, Dance Forms, and Known For
Andhra Pradesh has its capital in Amaravati. The state speaks Telugu and Urdu, and is famous for the classical Kuchipudi dance form. Andhra Pradesh is known for its rich cultural heritage, the Tirupati temple, and its spicy cuisine.
Arunachal Pradesh's capital is Itanagar. The state is home to many tribal languages, with Hindi and English also spoken. The state’s famous dance forms include Bihu and Popir, and it is known for its scenic beauty, the Tawang Monastery, and its tribal culture.
Assam, with its capital Dispur, primarily speaks Assamese and Bodo. Its dance forms like Bihu and Sattriya are famous. Assam is renowned for its tea gardens, the Kaziranga National Park, and the mighty Brahmaputra River.
Bihar has Patna as its capital. The main languages spoken are Hindi, Maithili, and Bhojpuri. The state is famous for the Bidesia dance and is known for its historical significance, including the ruins of Nalanda University and religious sites like Bodh Gaya.
Chhattisgarh's capital is Raipur, and the major languages spoken are Hindi and Chhattisgarhi. The state is known for its tribal dance forms like Raut Nacha and Panthi. Chhattisgarh is famous for the Bastar Dussehra festival and its forest industries.
Goa has Panaji as its capital. The official languages are Konkani and Marathi, and the dance forms like Fugdi and Dekhni are a cultural highlight. Goa is famous for its beaches, Portuguese architecture, and vibrant nightlife.
Gujarat's capital is Gandhinagar, with Gujarati as the main language. Famous for the Garba and Dandiya dance forms, Gujarat is known for its rich cultural heritage, the Rann of Kutch, and the Statue of Unity (Sardar Vallabhbhai Patel).
Haryana, with its capital in Chandigarh, speaks Hindi and Haryanvi. Popular dance forms in the state include Ghoomar and Dhamal. Haryana is known for its agricultural productivity, Kurukshetra, and wrestling traditions.
The capital of Himachal Pradesh is Shimla. The languages spoken include Hindi and Pahari. The famous dance form here is Nati. Himachal Pradesh is known for its hill stations, apple orchards, and trekking routes.
Jharkhand has Ranchi as its capital, with Hindi and Santali as major languages. The state is famous for dance forms like Chhau and Paika. Jharkhand is known for its dense forests, mining industry, and tribal communities.
Karnataka’s capital is Bengaluru, and Kannada is the main language. Karnataka is famous for Yakshagana and Bharatanatyam dance forms. The state is well-known as an IT hub, and for historical sites like Hampi and its coffee plantations.
Kerala has Thiruvananthapuram as its capital and Malayalam as the primary language. Kathakali and Mohiniyattam are famous dance forms in Kerala. The state is famous for its backwaters, Ayurvedic treatments, and temple festivals.
Madhya Pradesh's capital is Bhopal. The state speaks Hindi, and its dance forms include Lavani and Kalbelia. Madhya Pradesh is known for Khajuraho temples, Bandhavgarh National Park, and its historic forts.
Maharashtra has Mumbai as its capital. The primary language spoken is Marathi, and the state is known for dance forms like Lavani and Povada. Maharashtra is famous for Bollywood, the Gateway of India, and Ajanta-Ellora caves.
Manipur's capital is Imphal, and the language spoken is Manipuri. Manipuri Dance is one of the oldest classical dance forms in India. The state is famous for Loktak Lake, the Sangai Festival, and its rich cultural heritage.
Meghalaya has Shillong as its capital, with Khasi and Garo as main languages. The Shad Suk Mynsiem dance form is widely performed. Meghalaya is known for its living root bridges, being the cleanest state, and its natural beauty.
Mizoram's capital is Aizawl. Mizo is the main language. Cheraw is a traditional bamboo dance form here. Mizoram is famous for its lush green landscapes, traditional weaving, and bamboo dance.
Nagaland has Kohima as its capital, and Nagameses and English are widely spoken. The state is known for the Bambu Dance and the Hornbill Festival, which showcases vibrant tribal culture.
Odisha's capital is Bhubaneswar, and Oriya is the main language. The classical Odissi dance form originates here. Odisha is known for the Konark Sun Temple, Puri Jagannath Temple, and the Chilika Lake.
Punjab has Chandigarh as its capital, and Punjabi is the primary language. The state is famous for its Bhangra and Gidda dance forms. Punjab is known for the Golden Temple, its agriculture, and Amritsar.
Rajasthan's capital is Jaipur. The state speaks Hindi and Rajasthani. Ghoomar and Kalbelia are the famous dance forms here. Rajasthan is known for its palaces, forts, and the Thar Desert.
Sikkim has Gangtok as its capital. The state speaks Nepali and Sikkimese, and its famous dance form is the Mask Dance. Sikkim is known for the stunning Kanchenjunga, its Buddhist monasteries, and its lush green landscapes.
Tamil Nadu's capital is Chennai, and Tamil is the main language. The famous dance form is Bharatanatyam. Tamil Nadu is known for its temples, rich South Indian culture, and classical music.
Telangana has Hyderabad as its capital, and Telugu and Urdu are the main languages. The state is famous for the Kuchipudi dance form. Telangana is known for Charminar, being a major IT hub, and the city of Hyderabad as the Pearl City.
Tripura's capital is Agartala. The main languages spoken are Bengali and Kokborok, with Hojagiri being a famous dance form. Tripura is known for its royal palaces and natural beauty.
Uttar Pradesh's capital is Lucknow, and the main languages are Hindi and Urdu. The famous dance form is Kathak. Uttar Pradesh is known for the Taj Mahal, the holy city of Varanasi, and the Kumbh Mela.
Uttarakhand's capital is Dehradun. The languages spoken here are Hindi, Garhwali, and Kumaoni. The state is known for its dance form Langvir. Uttarakhand is famous for Haridwar, Nainital, and its stunning Himalayan range.
West Bengal has Kolkata as its capital, and the main language is Bengali. The state is famous for the Baul and Chhau dance forms. West Bengal is known for the Durga Puja, the Howrah Bridge, and its tea gardens."""
    question=get_audio_input()#this will call the function that will take the question from user in the form of speech
    #will check whether the question asked is valid or not
    if question:
        answer=generate_answer(question,context)
        print(f"Answer:{answer}")
        text_to_speech(answer)#will convert the answer generated in the form of text to speech.
if __name__=="__main__":
    main()