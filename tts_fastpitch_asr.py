import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from speechbrain.pretrained.interfaces import foreign_class
from speechbrain.pretrained import EncoderDecoderASR
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.models import FastPitchModel
import soundfile as sf

input_word = input("The english words you want to learn: ")
input_word = input_word.lower()
print("Input word: ", input_word)

# Translation
mode_name = 'liam168/trans-opus-mt-en-zh'
model_en2zh = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation = pipeline("translation_en_to_zh", model=model_en2zh, tokenizer=tokenizer)
meaning = translation(input_word, max_length=400)
meaning = meaning[0]["translation_text"]
print("The meaning of the word: ", meaning)

# TTS
# tts-tacotron2-ljspeech
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
mel_output, mel_length, alignment = tacotron2.encode_text(input_word)
waveforms = hifi_gan.decode_batch(mel_output)
signal = waveforms.squeeze(1)
torchaudio.save(f'{input_word}_tacotron2.wav', signal, 22050)

# tts_en_fastpitch
spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
model_tf = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
audio = model_tf.convert_spectrogram_to_audio(spec=spectrogram)
signal = audio.to('cpu').detach().numpy()[0]
sf.write(f'{input_word}_fastpitch.wav', signal, 22050)

# pause = input("Press Enter to continue")

tts_model = ["tacotron2", "fastpitch"]
output_dict = dict()
output_dict["input_word"] = input_word
output_dict["translation"] = meaning
output_dict["homophonic"] = []

# ASR
# asr-wav2vec2-ctc-aishell
asr_model = foreign_class(
    source="speechbrain/asr-wav2vec2-ctc-aishell",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderDecoderASR",
    # run_opts={"device":"cuda"}
)
# asr-transformer-aishell
asr_model2 = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-aishell", 
    savedir="pretrained_models/asr-transformer-aishell",
    # run_opts={"device":"cuda"}
)
# asr-wav2vec2-transformer-aishell
asr_model3 = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-transformer-aishell",
    savedir="pretrained_models/asr-wav2vec2-transformer-aishell",
    run_opts={"device":"cuda"}
)
for i, tts in enumerate(tts_model):
    asr_output = []
    # asr-wav2vec2-ctc-aishell
    text = asr_model.transcribe_file(f'{input_word}_{tts}.wav')
    text = "".join(text)
    text.replace(" ", "")
    print("asr-wav2vec2-ctc-aishell: ", text)
    asr_output.append(text)

    # asr-transformer-aishell
    text2 = asr_model2.transcribe_file(f'{input_word}_{tts}.wav')
    text2 = "".join(text2)
    text2.replace(" ", "")
    print("asr-transformer-aishell: ", text2)
    asr_output.append(text2)

    # asr-wav2vec2-transformer-aishell
    text3 = asr_model3.transcribe_file(f'{input_word}_{tts}.wav')
    text3 = "".join(text3)
    text3.replace(" ", "")
    print("asr-wav2vec2-transformer-aishell: ", text3)
    asr_output.append(text3)
    output_dict["homophonic"].append(asr_output)

print("Output: \n", output_dict)