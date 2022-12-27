from speechbrain.pretrained.interfaces import foreign_class
from speechbrain.pretrained import EncoderDecoderASR
from typing import List
import os


class AutomaticSpeechRecognition:
    TTS_MODEL = "tacotron2"
    ASR_SRC = "speechbrain/asr-wav2vec2-ctc-aishell"
    ASR_MODULE = "custom_interface.py"
    ASR_CLASSNAME = "CustomEncoderDecoderASR"

    ASR2_SRC = "speechbrain/asr-transformer-aishell"
    ASR2_SAVE = "pretrained_models/asr-transformer-aishell"

    ASR3_SRC = "speechbrain/asr-wav2vec2-transformer-aishell"
    ASR3_SAVE = "pretrained_models/asr-wav2vec2-transformer-aishell"

    def __init__(
        self,
        tmp_save_dir: str = "tmp",
    ) -> None:
        self.tmp_save_dir = tmp_save_dir

        print("Downloading ASR model 1: %s..." % self.ASR_SRC)
        self.asr_model = foreign_class(
            source=self.ASR_SRC,
            pymodule_file=self.ASR_MODULE,
            classname=self.ASR_CLASSNAME,
        )
        print("Downloading ASR model 2: %s..." % self.ASR2_SRC)
        self.asr_model2 = EncoderDecoderASR.from_hparams(
            source=self.ASR2_SRC,
            savedir=self.ASR2_SAVE,
        )
        print("Downloading ASR model 3: %s..." % self.ASR3_SRC)
        self.asr_model3 = EncoderDecoderASR.from_hparams(
            source=self.ASR3_SRC,
            savedir=self.ASR3_SAVE,
            run_opts={"device": "cuda"},
        )

    def recognize(self, word: str) -> List[str]:
        def helper(model, word: str) -> str:
            text = model.transcribe_file(os.path.join(self.tmp_save_dir, "%s_%s.wav" % (word, self.TTS_MODEL)))
            text = "".join(text)
            text.replace(" ", "")
            return text

        homophonic = []
        homophonic.append(helper(self.asr_model, word))
        homophonic.append(helper(self.asr_model2, word))
        homophonic.append(helper(self.asr_model3, word))

        return homophonic
