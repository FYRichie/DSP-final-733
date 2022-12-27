from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio
import os


class TextToSpeech:
    MODEL_NAME = "liam168/trans-opus-mt-en-zh"
    TASK = "translation_en_to_zh"
    TACOTRON = "speechbrain/tts-tacotron2-ljspeech"
    HIFI_GAN = "speechbrain/tts-hifigan-ljspeech"

    TACO_SAVE = "tmpdir_tts"
    HIFI_SAVE = "tmpdir_vocoder"

    def __init__(
        self,
        max_length: int = 400,
        tmp_save_dir: str = "tmp",
    ) -> None:
        self.model_name = self.MODEL_NAME
        self.pipeline_task = self.TASK
        self.max_length = max_length
        self.tmp_save_dir = tmp_save_dir
        if not os.path.isdir(tmp_save_dir):
            os.mkdir(tmp_save_dir)

        print("Downloading translation model %s..." % self.MODEL_NAME)
        self.model_en2zh = AutoModelWithLMHead.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        print("Downloading text-to-speech model %s and %s..." % (self.TACOTRON, self.HIFI_GAN))
        self.tacotron2 = Tacotron2.from_hparams(source=self.TACOTRON, savedir=self.TACO_SAVE)
        self.hifi_gan = HIFIGAN.from_hparams(source=self.HIFI_GAN, savedir=self.HIFI_SAVE)

        print("Creating pipeline %s..." % self.TASK)
        self.translator = pipeline(task=self.TASK, model=self.model_en2zh, tokenizer=self.tokenizer)

    def translate(self, word: str) -> str:
        print("Translating %s to chinese..." % word)
        meaning = self.translator(word, max_length=self.max_length)

        trans = ""
        for c in meaning[0]["translation_text"]:
            if c not in trans:
                trans += c
        return trans

    def to_speech(self, word: str) -> None:
        print("Turning %s into speech..." % word)
        mel_output, mel_length, alignment = self.tacotron2.encode_text(word)
        waveforms = self.hifi_gan.decode_batch(mel_output)
        signal = waveforms.squeeze(1)
        torchaudio.save(os.path.join(self.tmp_save_dir, "%s_tacotron2.wav" % word), signal, 22050)
