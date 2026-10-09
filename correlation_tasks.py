
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    filepath = getattr(data, 'file_path', getattr(data, 'path', str(data)))

    df = pd.read_csv(filepath, sep='\t', na_values=['NA'])

    male_df = df[df['Gender'] == 'Male']
    female_df = df[df['Gender'] == 'Female']

    target = 'MRI_Count'
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    male_correlations = male_df[features + [target]].corr()[target][features].to_dict()
    female_correlations = female_df[features + [target]].corr()[target][features].to_dict()

    strongest_mri_feature = None
    max_abs_corr = -1.0

    for feature in features:
        if abs(male_correlations[feature]) > max_abs_corr:
            max_abs_corr = abs(male_correlations[feature])
            strongest_mri_feature = feature

        if abs(female_correlations[feature]) > max_abs_corr:
            max_abs_corr = abs(female_correlations[feature])
            strongest_mri_feature = feature

    return BrainCorrelationSummary(
        male_correlations = male_correlations,
        female_correlations = female_correlations,
        strongest_mri_feature = strongest_mri_feature
    )