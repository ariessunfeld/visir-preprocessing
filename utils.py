from pathlib import Path
from typing import Union
import pandas as pd

def correct_spectral_offsets(
    file_path: Union[str, Path],
    join1: tuple = (1000, 1001),
    join2: tuple = (1800, 1801)
) -> pd.DataFrame:
    """
    Corrects step discontinuities in spectral data from ASD .txt files.

    Parameters:
        file_path (str or Path): Path to the .asd.txt file.
        join1 (tuple): First join wavelengths (e.g., between VNIR and SWIR1).
        join2 (tuple): Second join wavelengths (e.g., between SWIR1 and SWIR2).

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y'] of the corrected spectrum.
    """
    file_path = Path(file_path)
    spectral_data = []

    with file_path.open('r') as file:
        for line in file:
            if "\t" in line:
                try:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        spectral_data.append((x_val, y_val))
                except ValueError:
                    continue

    df = pd.DataFrame(spectral_data, columns=["x", "y"])

    # Step 1: First join correction
    y1 = df.loc[df["x"] == join1[0], "y"].values[0]
    y2 = df.loc[df["x"] == join1[1], "y"].values[0]
    d1 = y2 - y1

    seg1 = df[df["x"] <= join1[0]]
    seg2 = df[(df["x"] >= join1[1]) & (df["x"] <= join2[0])].copy()
    seg2["y"] = seg2["y"] - d1

    transformed_array = pd.concat([seg1, seg2])

    # Step 2: Second join correction
    y3 = transformed_array.loc[transformed_array["x"] == join2[0], "y"].values[0]
    y4 = df.loc[df["x"] == join2[1], "y"].values[0]
    d2 = y4 - y3

    seg3 = df[df["x"] > join2[0]].copy()
    seg3["y"] = seg3["y"] - d2

    final_array = pd.concat([transformed_array, seg3])
    return final_array
