
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    filepath = getattr(data, 'file_path', getattr(data, 'path', str(data)))

    df = pd.read_csv(data.csv_path)

    missing_counts = df.isna().sum().to_dict()

    over_30_count = int((df['Age'] > 30).sum())

    mean_age = df.groupby('Pclass')['Age'].median().to_dict()

    survival_rate = df.groupby('Pclass')['Survived'].mean().to_dict()

    top_5_fares = df['Fare'].nlargest(5).tolist()

    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_counts,
        adults_over_30_count=over_30_count,
        mean_age_by_pclass=mean_age,
        survival_rate_by_pclass=survival_rate,
        highest_fares=top_5_fares,
    )

if __name__ == "__main__":
    result = analyze_titanic(TitanicInput(csv_path="titanic.csv"))
    print("Строк:", result.row_count)
    print("Пропуски:", result.missing_by_column)
    print("Старше 30:", result.adults_over_30_count)
    print("Средний возраст по классам:", result.mean_age_by_pclass)
    print("Доля выживших по классам:", result.survival_rate_by_pclass)
    print("Топ-5 Fare:", result.highest_fares)