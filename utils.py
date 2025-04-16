from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from astropy.io import fits

def extract_spectrum_from_fits(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Extracts wavelength and reflectance data from a SCAM .fits file.

    Parameters:
        file_path (str or Path): Path to the FITS file.

    Returns:
        pd.DataFrame: DataFrame with 'x' (wavelength in µm) and 'y' (reflectance) columns.
    """
    file_path = Path(file_path)
    
    with fits.open(file_path) as hdul:
        reflectance = hdul['Spectra'].data['I_F_Atm']
        wavelength = hdul['Wavelength'].data['Wavelength (um)']
    
    return pd.DataFrame({'x': wavelength, 'y': reflectance})

def extract_spectrum_from_txt(
    file_path: Union[str, Path],
    correct_offsets: bool = False,
    **offset_params
) -> pd.DataFrame:
    """
    Extracts spectral data from an ASD .txt file, with optional offset correction.

    Parameters:
        file_path (str or Path): Path to the .asd.txt file.
        correct_offsets (bool): Whether to apply join correction for spectrometer offsets.
        **offset_params: Additional join-related kwargs passed to the correction function (e.g., join1, join2).

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y'].
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

    if correct_offsets:
        # Apply offset correction if requested
        join1 = offset_params.get("join1", (1000, 1001))
        join2 = offset_params.get("join2", (1800, 1801))

        # Compute first join offset
        y1 = df.loc[df["x"] == join1[0], "y"].values[0]
        y2 = df.loc[df["x"] == join1[1], "y"].values[0]
        d1 = y2 - y1

        seg1 = df[df["x"] <= join1[0]]
        seg2 = df[(df["x"] >= join1[1]) & (df["x"] <= join2[0])].copy()
        seg2["y"] = seg2["y"] - d1
        transformed_array = pd.concat([seg1, seg2])

        # Compute second join offset
        y3 = transformed_array.loc[transformed_array["x"] == join2[0], "y"].values[0]
        y4 = df.loc[df["x"] == join2[1], "y"].values[0]
        d2 = y4 - y3

        seg3 = df[df["x"] > join2[0]].copy()
        seg3["y"] = seg3["y"] - d2

        return pd.concat([transformed_array, seg3])

    return df

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

def preprocess_spectral_folder(
    input_folder: Union[str, Path],
    overwrite: bool = False,
    correct_txt_offsets: bool = True,
    **offset_params
) -> None:
    """
    Preprocesses all .fits and .asd.txt spectral files in a folder, converting them to standardized CSVs.

    Parameters:
        input_folder (str or Path): Folder containing raw .fits and .txt files (non-recursive).
        overwrite (bool): Whether to overwrite existing CSV output files.
        correct_txt_offsets (bool): Whether to apply offset correction to .txt files.
        **offset_params: Additional parameters for offset correction (e.g., join1, join2).
    """
    input_folder = Path(input_folder)
    output_folder = input_folder.parent / f"{input_folder.name}_processed"
    output_folder.mkdir(exist_ok=True)

    for file in input_folder.iterdir():
        if file.suffix == '.fits' or file.suffix == '.txt':
            output_csv = output_folder / f"{file.stem}.csv"
            if output_csv.exists() and not overwrite:
                print(f"Warning: Output file already exists, will NOT overwrite: {output_csv.name} (use overwrite=True to overwrite)")
                continue  # Warn and skip
            try:
                if file.suffix == '.fits':
                    df = extract_spectrum_from_fits(file)
                elif file.suffix == '.txt':
                    df = extract_spectrum_from_txt(file, correct_offsets=correct_txt_offsets, **offset_params)
                df.to_csv(output_csv, index=False)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
        else:
            print(f'Skipping file {file.name} (unrecognized suffix: {file.suffix})')
            continue

