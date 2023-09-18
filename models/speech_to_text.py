import csv
import pandas as pd
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from typing import Iterable


# 音声ファイルからテキストデータを作成するためのクラス
class SpeechToText:
    # 学習済みモデルをクラス変数として保持（メモリ節約のため）
    learned_model: WhisperModel = None

    def __init__(self, audio_file_path: str):
        # 入力する音声ファイルのパス
        self.audio_file_path = audio_file_path
        # 文字起こしの結果を格納する変数
        self.segments: Iterable[Segment] = None

        # 学習済みモデルが未設定の場合はエラーを発生する
        if not self.learned_model:
            raise Exception("クラス変数'learned_model'が未設定です")

    # 引数の秒数（数値）を"00:00:00"形式の文字列に変換する
    def __convert_seconds_to_hms(self, arg_seconds: int) -> str:
        hours = arg_seconds // 3600
        minutes = (arg_seconds % 3600) // 60
        seconds = arg_seconds % 60

        return f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"

    # 学習済みモデルを設定する
    @classmethod
    def set_learned_model(
        cls, model_size: str, device: str = "cpu", compute_type: str = "int8"
    ):
        # 学習済みモデルを読み込む
        cls.learned_model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    # 文字起こしを実行する（initial_promptは、ヒントとなる文字列）
    def transcribe(self, language: str = "ja", initial_prompt: str | None = None):
        # 音声ファイルを読み込み、文字起こしを実行する
        self.segments, _ = self.learned_model.transcribe(
            self.audio_file_path, language=language, initial_prompt=initial_prompt
        )

    # 文字起こしの結果をCSVファイルに書き込む
    def write_to_csv(self, csv_file_path: str, encoding="utf_8_sig"):
        # 文字起こしの結果（オブジェクト）をDataFrameに変換する
        df_segments = pd.DataFrame(self.segments)
        COLUMNS_TIME = ["start", "end"]

        # 不要な列を削除する
        df_segments.drop(
            columns=[
                "seek",
                "tokens",
                "temperature",
                "avg_logprob",
                "compression_ratio",
                "no_speech_prob",
                "words",
            ],
            inplace=True,
        )
        # "start"と"end"の列定義を数値から文字列に変更する
        df_segments[COLUMNS_TIME] = df_segments[COLUMNS_TIME].astype(str)

        # 各行のstartとendを秒数から"00:00:00"形式の文字列に変換する
        for index, row in df_segments.iterrows():
            for column in COLUMNS_TIME:
                # 秒数→"00:00:00"
                hms = self.__convert_seconds_to_hms(int(row[column]))
                # 変換した値で上書きする
                df_segments.loc[index, column] = hms

        # CSVファイルを出力する
        df_segments.to_csv(
            csv_file_path, index=False, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC
        )
