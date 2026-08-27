import requests
import pandas as pd
import os

TCGA_PATIENT_IDS = "metadata/TCGA_patient_list.txt"
OUTPUT_DIR = "data/TCGA/msi_data"
CANCER_TYPES = ["COAD", "UCEC", "STAD"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_tcga_barcode(barcode):
    """
    Parse TCGA barcode and extract different levels.
    
    TCGA barcode format: TCGA-XX-XXXX-XXX-XXX-XXXX-XX
    Example: TCGA-3L-AA1B-01A-11D-A40W-09
    
    Parts:
    - Project: TCGA
    - TSS: 3L (Tissue Source Site)
    - Participant: AA1B
    - Sample: 01A (01 = Primary Solid Tumor, A = vial)
    - Portion: 11D
    - Plate: A40W
    - Center: 09
    """
    parts = barcode.split('-')
    
    result = {
        'full_barcode': barcode,
        'patient_barcode': None,
        'sample_barcode': None,
        'sample_type': None
    }
    
    if len(parts) >= 3:
        # Patient barcode: TCGA-XX-XXXX
        result['patient_barcode'] = '-'.join(parts[:3])
    
    if len(parts) >= 4:
        # Sample barcode: TCGA-XX-XXXX-XX
        result['sample_barcode'] = '-'.join(parts[:4])
        
        # Sample type code (01 = Primary Tumor, 10 = Blood, 11 = Normal, etc.)
        sample_code = parts[3][:2]
        result['sample_type'] = sample_code
    
    return result


def get_tcga_studies_list():
    """
    Get list of all TCGA studies from cBioPortal.
    """
    base_url = "https://www.cbioportal.org/api"
    url = f"{base_url}/studies"
    
    headers = {"accept": "application/json"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        studies = response.json()
        tcga_studies = [s for s in studies if 'tcga' in s['studyId'].lower()]
        return tcga_studies
    return []


def get_msi_data_cbioportal(study_id):
    """
    Get MSI data from cBioPortal and convert to GDC-compatible format.
    """
    base_url = "https://www.cbioportal.org/api"
    
    print(f"\nFetching data for study: {study_id}")
    
    # Get clinical data
    url = f"{base_url}/studies/{study_id}/clinical-data"
    params = {"clinicalDataType": "SAMPLE"}
    headers = {"accept": "application/json"}
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    clinical_data = response.json()
    
    if not clinical_data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(clinical_data)
    
    # Pivot to wide format
    clinical_wide = df.pivot_table(
        index='sampleId',
        columns='clinicalAttributeId',
        values='value',
        aggfunc='first'
    )
    
    # Parse TCGA barcodes
    barcode_info = []
    for sample_id in clinical_wide.index:
        parsed = parse_tcga_barcode(sample_id)
        parsed['original_sample_id'] = sample_id
        barcode_info.append(parsed)
    
    barcode_df = pd.DataFrame(barcode_info)
    barcode_df.set_index('original_sample_id', inplace=True)
    
    # Merge with clinical data
    result = clinical_wide.join(barcode_df)
    
    return result


def get_all_tcga_msi_data():
    """
    Get MSI data from all TCGA studies in cBioPortal.
    """
    print("Fetching TCGA studies from cBioPortal...")
    studies = get_tcga_studies_list()
    
    print(f"Found {len(studies)} TCGA studies")
    
    all_data = []
    
    for study in studies:
        study_id = study['studyId']
        study_name = study.get('name', '')
        
        # Skip non-TCGA or provisional studies if you want only final versions
        if 'provisional' in study_id.lower():
            continue
        
        print(f"\nProcessing: {study_id}")
        
        data = get_msi_data_cbioportal(study_id)
        
        if data is not None:
            # Look for MSI columns
            msi_cols = [col for col in data.columns if 'msi' in str(col).lower()]
            
            if msi_cols:
                print(f"Found MSI data: {msi_cols}")
                data['study_id'] = study_id
                data['study_name'] = study_name
                all_data.append(data)
            else:
                print(f"No MSI data found")
    
    if all_data:
        combined = pd.concat(all_data, axis=0)
        return combined
    
    return None


def get_msi_for_specific_cancer(cancer_type="COAD"):
    """
    Get MSI data for a specific TCGA cancer type.
    """
    print(f"Searching for TCGA-{cancer_type} studies...")
    studies = get_tcga_studies_list()
    
    # Find matching studies
    matching = [s for s in studies if cancer_type.lower() in s['studyId'].lower()]
    
    if not matching:
        print(f"No studies found for {cancer_type}")
        return None
    
    print(f"\nFound {len(matching)} matching studies:")
    for s in matching:
        print(f"  - {s['studyId']}: {s.get('name', '')}")
    
    # Prefer PanCancer Atlas or pub versions over provisional
    preferred = [s for s in matching if 'pan_can' in s['studyId'] or 'pub' in s['studyId']]
    selected = preferred[0] if preferred else matching[0]
    
    print(f"\nUsing: {selected['studyId']}")
    
    data = get_msi_data_cbioportal(selected['studyId'])
    
    if data is not None:
        # Show what we got
        print(f"\nRetrieved data for {len(data)} samples")
        print(f"Columns: {len(data.columns)}")
        
        # Look for MSI columns
        msi_cols = [col for col in data.columns if 'msi' in str(col).lower()]
        
        if msi_cols:
            print(f"\nMSI-related columns: {msi_cols}")
            for col in msi_cols:
                print(f"\n{col} distribution:")
                print(data[col].value_counts())
        
        # Show barcode parsing results
        if 'patient_barcode' in data.columns:
            print(f"\nSample barcode formats:")
            print(f"  Full barcode example: {data.index[0]}")
            print(f"  Patient barcode example: {data['patient_barcode'].iloc[0]}")
            if 'sample_barcode' in data.columns:
                print(f"  Sample barcode example: {data['sample_barcode'].iloc[0]}")
    
    return data


def match_msi_to_gdc_samples(msi_data, gdc_sample_ids):
    """
    Match MSI data from cBioPortal to your GDC sample IDs.
    
    Parameters:
    - msi_data: DataFrame from cBioPortal with MSI data
    - gdc_sample_ids: List of your GDC sample IDs (e.g., ['TCGA-A3-A8OW',...])
    
    Returns:
    - DataFrame with MSI data matched to your sample IDs
    """
    if 'patient_barcode' not in msi_data.columns:
        print("Error: patient_barcode not found in data")
        return None
    
    # Create mapping from patient barcode to MSI data
    # Group by patient barcode (in case multiple samples per patient)
    msi_cols = [col for col in msi_data.columns if 'msi' in str(col).lower()]
    
    if not msi_cols:
        print("No MSI columns found")
        return None
    
    # For each patient, take the first sample's MSI status
    # (usually consistent across samples from same patient)
    patient_msi = msi_data.groupby('patient_barcode')[msi_cols].first()
    
    # Match to your GDC sample IDs
    matched_data = []
    
    for sample_id in gdc_sample_ids:
        # Your GDC IDs might be patient-level (TCGA-A3-A8OW)
        # or sample-level (TCGA-A3-A8OW-01)
        
        # Extract patient barcode from your ID
        parts = sample_id.split('-')
        if len(parts) >= 3:
            patient_barcode = '-'.join(parts[:3])
            
            if patient_barcode in patient_msi.index:
                row = patient_msi.loc[patient_barcode].to_dict()
                row['gdc_sample_id'] = sample_id
                row['patient_barcode'] = patient_barcode
                matched_data.append(row)
            else:
                # No MSI data for this patient
                matched_data.append({
                    'gdc_sample_id': sample_id,
                    'patient_barcode': patient_barcode,
                    **{col: None for col in msi_cols}
                })
    
    result_df = pd.DataFrame(matched_data)
    
    print(f"\nMatching results:")
    print(f"  Total GDC samples: {len(gdc_sample_ids)}")
    print(f"  Samples with MSI data: {result_df[msi_cols[0]].notna().sum()}")
    print(f"  Samples without MSI data: {result_df[msi_cols[0]].isna().sum()}")
    
    return result_df


def save_cancer_msi_data(cancer_type, gdc_ids):
    """Download, simplify, match, and save MSI/MSS data for one TCGA cancer."""
    cancer_slug = cancer_type.lower()

    print("\n" + "=" * 80)
    print(f"Getting MSI data for TCGA-{cancer_type}")
    print("=" * 80)

    msi_data = get_msi_for_specific_cancer(cancer_type)
    if msi_data is None:
        print(f"No MSI data retrieved for TCGA-{cancer_type}")
        return

    full_path = os.path.join(OUTPUT_DIR, f"tcga_{cancer_slug}_msi_full.csv")
    msi_data.to_csv(full_path, index=False)
    print(f"\nFull data saved to {full_path}")

    msi_cols = [col for col in msi_data.columns if 'msi' in str(col).lower()]
    if not msi_cols or 'patient_barcode' not in msi_data.columns:
        print(f"No usable MSI/MSS columns found for TCGA-{cancer_type}")
        return

    simplified_cols = ['patient_barcode']
    if 'sample_barcode' in msi_data.columns:
        simplified_cols.append('sample_barcode')
    simplified_cols.extend(msi_cols)

    simplified = msi_data[simplified_cols].copy()
    simplified = simplified.dropna(subset=msi_cols, how='all')

    simplified_path = os.path.join(
        OUTPUT_DIR,
        f"tcga_{cancer_slug}_msi_simplified.csv",
    )
    simplified.to_csv(simplified_path, index=False)
    print(f"Simplified data saved to {simplified_path}")
    print(f"\nSimplified data preview:")
    print(simplified.head(10))

    print("\n" + "=" * 80)
    print(f"Matching TCGA-{cancer_type} MSI data to GDC sample IDs")
    print("=" * 80)

    matched = match_msi_to_gdc_samples(msi_data, gdc_ids)
    if matched is not None:
        matched_path = os.path.join(
            OUTPUT_DIR,
            f"tcga_{cancer_slug}_msi_matched.csv",
        )
        matched.to_csv(matched_path, index=False)
        print(f"\nMatched data saved to {matched_path}")
        print(matched)


if __name__ == "__main__":
    gdc_ids = pd.read_csv(TCGA_PATIENT_IDS, header=None)[0].tolist()

    for cancer_type in CANCER_TYPES:
        save_cancer_msi_data(cancer_type, gdc_ids)
