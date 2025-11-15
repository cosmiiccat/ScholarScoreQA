import json
import ast
import pandas as pd
import os

def fmt_encode(row):
    answer_json = ast.literal_eval(json.loads(row.to_json())["Answer"])
    return "\n".join(
        f"{key}:\n  {value if value else 'None'}" if isinstance(value, list) and not value
        else f"{key}:\n  - " + "\n  - ".join(map(str, value)) if isinstance(value, list)
        else f"{key}:\n  {value}" for key, value in answer_json.items()
    )

def calc_evidence_percentage(row, evidence_type):
    try:
        percent = 0
        answer_data = ast.literal_eval(json.loads(row.to_json())["Answer"])
        evidence_list = answer_data[evidence_type]
        context = json.loads(row.to_json())["Context"]
        for evidence in evidence_list:
            if evidence in context:
                percent += 1
        return (percent / len(evidence_list)) * 100 if evidence_list else 0
    except:
        return 0

def process_dataframe(df):
    for idx, row in df.iterrows():
        try:
            df.loc[idx, "Flatten Answer"] = fmt_encode(row)
            df.loc[idx, "Evidence Percentage"] = calc_evidence_percentage(row, "evidence")
            df.loc[idx, "Highlighted Evidence Percentage"] = calc_evidence_percentage(row, "highlighted_evidence")
        except Exception as e:
            print(f"Error: at idx:{idx}, error_msg:{e}")
            continue
    return df

def main():
    data_path = os.getenv("DATA_PATH", "/content/drive/MyDrive/Data/Qasper/qasper_train_updated.csv")
    df_qasper_preprocessed = pd.read_csv(data_path)
    df_qasper_preprocessed = process_dataframe(df_qasper_preprocessed)
    return df_qasper_preprocessed.head(5)

if __name__ == "__main__":
    result = main()
    print(result)