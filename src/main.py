from translation import TextToSpeech
from recognition import AutomaticSpeechRecognition
from creation import SequenceCreation

if __name__ == "__main__":
    tts = TextToSpeech()
    asr = AutomaticSpeechRecognition()
    sc = SequenceCreation()

    while True:
        try:
            input_word = input("The english words you want to learn: ")

            # text to speech
            translation = tts.translate(input_word)
            print("The chinese translation of the word:", translation)
            tts.to_speech(input_word)

            # automatic speech recognition
            homophonic = asr.recognize(input_word)
            print("Homophonics:", homophonic)

            key_words = [translation + homo for homo in list(set(homophonic))]

            seqs = sc.get_sequence(key_words)
            print("Generated sequences:")
            for i, seq in enumerate(seqs):
                print("%d: %s" % (i, seq))
        except KeyboardInterrupt:
            print("\nExiting the process...")
            exit()
