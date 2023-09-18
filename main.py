from datetime import datetime

# オリジナルのクラスをインポート
from models.speech_to_text import SpeechToText


if __name__ == "__main__":
    MODEL_SIZE = "large-v2"
    DIR_PATHS = {"input": "./assets/input", "output": "./assets/output"}
    file_paths = {"audio": "", "csv": ""}
    speech_to_text: SpeechToText = None
    prefix = ""
    answer = ""

    # 入力ファイルの名前はユーザーが指定する
    answer = input("音声ファイルの名前を入力してください: ")
    file_paths["audio"] = f"{DIR_PATHS['input']}/{answer}"

    # 学習済みモデルを設定する
    SpeechToText.set_learned_model(MODEL_SIZE)
    speech_to_text = SpeechToText(file_paths["audio"])
    answer = input("文字起こしのヒントとなる文字列を入力してください（省略可）: ")
    # 文字起こしを実行する
    speech_to_text.transcribe(initial_prompt=answer if answer else None)

    # 出力ファイルの接頭辞を現在時刻から生成する
    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_paths["csv"] = f"{DIR_PATHS['output']}/{prefix}_output.csv"
    # 文字起こしの結果をCSVファイルに書き込む
    speech_to_text.write_to_csv(file_paths["csv"])
