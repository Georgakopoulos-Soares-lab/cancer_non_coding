import os
import pandas as pd

# input files/dir
DRIVER_GENES_INTOGEN = "data/driver_genes_intogen_curated"
SAMPLE_SHEET = "data/TCGA/metadata/gdc_sample_sheet.2025-11-04.tsv"
GENCODE_ANNOT = "data/ref/gencode.v38.annotation.gtf"
CHROM_SIZES = "data/ref/hg38.chrom.sizes"

# output files/dir
DRIVER_GENES_DIR = "data/driver_genes_coords"
os.makedirs(DRIVER_GENES_DIR, exist_ok=True)
GENE_LIST = "metadata/tcga_gene_list.tsv"

# get list of cancer types from the cancer samples
samples_df = pd.read_csv(SAMPLE_SHEET, sep="\t")
CANCER_TYPES = samples_df["Project ID"].unique().tolist()
CANCER_TYPES = [cancer.replace("TCGA-", "") for cancer in CANCER_TYPES]
CANCER_TYPES.sort()
print(f"Found {len(CANCER_TYPES)} cancer types:", CANCER_TYPES)
CANCER_TYPES = CANCER_TYPES + ["Pancancer"] # add pancancer gene list

# get dictionary of chromosome sizes
chrom_sizes = pd.read_csv(CHROM_SIZES, sep="\t", header=None, names=["chrom", "size"])
chrom_sizes["size"] = chrom_sizes["size"].astype(int)
chrom_sizes_dict = chrom_sizes.set_index("chrom")["size"].to_dict()

# gene annotation from gencode for getting gene start and end positions
gencode_annot = pd.read_csv(GENCODE_ANNOT, sep="\t", comment="#", header=None)
gencode_annot = gencode_annot[gencode_annot[2] == "gene"]
gencode_annot = gencode_annot.sort_values(by=[0, 3, 4]).reset_index(drop=True)

def get_extended_coords(coords, length, prev_coord, next_coord):
	"""
	coords: tuple (chrom, start, end) for the gene
	strand: '+' or '-'
	upstream_length, downstream_length: lengths in bp
	prev_coord, next_coord: coordinates of previous and next genes (chrom, start, end) or None
	chrom_sizes_dict: dict of chromosome lengths
	Returns: (chrom, extended_start, extended_end)
	"""
	chrom, gene_start, gene_end = coords
	# Upstream region
	start = max(1, gene_start - length) # ensure start is at least 1
	if prev_coord is not None and start <= prev_coord[2]: # previous gene overlaps with upstream region
		start = prev_coord[2] + 1  # upstream region starts after end of previous gene
		if start > gene_start:
			start = gene_start
	# Downstream region
	end = min(gene_end + length, chrom_sizes_dict[chrom]) # ensure end does not exceed chromosome size
	if next_coord is not None and end >= next_coord[1]: # next gene overlaps with downstream region
		end = next_coord[1] - 1  # downstream region ends before start of next gene
		if end < gene_end:
			end = gene_end

	return (chrom, start, end)

def get_gene_and_neighbors(driver_gene, gencode_annot):
	"""
	Get gene coordinates and strand-aware previous and next gene coordinates.

	Parameters
	----------
	driver_gene : str
		Name of the gene to retrieve.
	gencode_annot : pd.DataFrame
		GTF annotation as a DataFrame with standard GTF columns:
		0=chrom, 2=feature, 3=start, 4=end, 6=strand, 8=attributes

	Returns
	-------
	dict
		{
			"gene_coords": (chrom, start, end),
			"strand": '+' or '-',
			"prev_coord": (chrom, start, end) or None,
			"next_coord": (chrom, start, end) or None
		}
	"""

	# Find the gene row
	gene_rows = gencode_annot[
		(gencode_annot[2] == "gene") &
		gencode_annot[8].str.contains(f'gene_name "{driver_gene}"')
	]
	if gene_rows.empty:
		print(f"Gene {driver_gene} not found in the annotation file")
		return None

	gene_of_interest = gene_rows.iloc[0]
	strand = gene_of_interest[6]
	gene_coords = (gene_of_interest[0], gene_of_interest[3], gene_of_interest[4])
	chrom = gene_of_interest[0]

	# All genes on the same chromosome
	genes_same_chrom = gencode_annot[
		(gencode_annot[0] == chrom) & (gencode_annot[2] == "gene")
	].sort_values(3).reset_index(drop=True)

	prev_genes = genes_same_chrom[genes_same_chrom[4] < gene_coords[1]]
	prev_coord = (prev_genes.iloc[-1][0], prev_genes.iloc[-1][3], prev_genes.iloc[-1][4]) if not prev_genes.empty else None

	next_genes = genes_same_chrom[genes_same_chrom[3] > gene_coords[2]]
	next_coord = (next_genes.iloc[0][0], next_genes.iloc[0][3], next_genes.iloc[0][4]) if not next_genes.empty else None

	return {
		"gene_coords": gene_coords,
		"strand": strand,
		"prev_coord": prev_coord,
		"next_coord": next_coord
	}

for cancer_type in CANCER_TYPES:
	cancer_type = cancer_type.replace("TCGA_", "")
	files = os.listdir(DRIVER_GENES_INTOGEN)
	if cancer_type == "Pancancer":
		# get genes muated in atleast 1% of samples across all cancer types
		driver_genes_intogen = pd.read_csv(f"{DRIVER_GENES_INTOGEN}/Pancancer.tsv", sep="\t")
		# driver_genes_intogen = driver_genes_intogen[driver_genes_intogen["Samples (%)"] >= 0.2]
		driver_gene_list = driver_genes_intogen["Symbol"].to_list()
	else:
		cancer_files = [f for f in files if f.startswith(cancer_type)]
		driver_genes_intogen = pd.DataFrame()
		for fname in cancer_files:
			driver_genes_intogen = pd.concat([driver_genes_intogen, pd.read_csv(f"{DRIVER_GENES_INTOGEN}/{fname}", sep="\t")])
		driver_gene_list = driver_genes_intogen["Symbol"].to_list()
		# TERT gene is known to be a driver in several cancers (cohorts selected according to PCAWG)
		tert_promoter_cancers = ['BLCA', 'GBM', 'HNSC', 'LIHC', 'SKCM', 'THCA']
		if "TERT" not in driver_gene_list and cancer_type in tert_promoter_cancers:
			driver_gene_list.append("TERT")

	gene_objects = []
	for driver_gene in driver_gene_list:
		gene_and_neighbors = get_gene_and_neighbors(driver_gene, gencode_annot)
		if gene_and_neighbors is None:
			continue
		gene_of_interest_coords = gene_and_neighbors["gene_coords"]
		strand = gene_and_neighbors["strand"]
		prev_coord = gene_and_neighbors["prev_coord"]
		next_coord = gene_and_neighbors["next_coord"]

		# coordinates of the driver genes including 10kB upstream and downstream of the gene, and promoter region
		extended_long_coords = get_extended_coords(gene_of_interest_coords, 10000, prev_coord, next_coord)
		extended_coords = get_extended_coords(gene_of_interest_coords, 2000, prev_coord, next_coord)
		gene_coord_obj = {
			"gene": driver_gene,
			"chr": gene_of_interest_coords[0],
			"strand": strand,
			"start": gene_of_interest_coords[1], # start of the gene
			"end": gene_of_interest_coords[2], # end of the gene
			"length": abs(gene_of_interest_coords[2] - gene_of_interest_coords[1]) + 1, # length of the gene
			"extended_start": extended_coords[1], # start of 2kb upstream of the gene
			"extended_end": extended_coords[2], # end of 2kb downstream of the gene 
			"extended_start_long": extended_long_coords[1], # start of 10kb upstream of the gene
			"extended_end_long": extended_long_coords[2], # end of 10kb downstream of the gene
		}
		gene_objects.append(gene_coord_obj)
	genes_df = pd.DataFrame(gene_objects)
	print(f"{cancer_type}: {len(genes_df)} driver genes")

	driver_genes_list = genes_df["gene"].tolist()
	genes_df.to_csv(f"{DRIVER_GENES_DIR}/{cancer_type}.tsv", sep="\t", index=False)

# get list of all driver genes and their coordinates
all_driver_genes = []
all_driver_coords = []
for cancer_type in CANCER_TYPES:
	cancer_type = cancer_type.replace("TCGA_", "")
	driver_genes_df = pd.read_csv(f"{DRIVER_GENES_DIR}/{cancer_type}.tsv", sep="\t")
	all_driver_genes.extend(driver_genes_df["gene"].tolist())
	all_driver_coords.extend(driver_genes_df[driver_genes_df["gene"].isin(all_driver_genes)].to_dict(orient="records"))

# number of cancer types associated with each gene driver
gene_cancer_counts = {}
for cancer_type in CANCER_TYPES:
	cancer_type = cancer_type.replace("TCGA_", "")
	driver_genes_df = pd.read_csv(f"{DRIVER_GENES_DIR}/{cancer_type}.tsv", sep="\t")
	for gene in driver_genes_df["gene"].tolist():
		gene_cancer_counts[gene] = gene_cancer_counts.get(gene, 0) + 1
all_driver_genes_df = pd.DataFrame({
	"gene": list(gene_cancer_counts.keys()),
	"num_cancer_types": list(gene_cancer_counts.values())
})
all_driver_genes_df.to_csv(GENE_LIST, sep="\t", index=False)

# all driver gene coordinates
all_driver_coords_df = pd.DataFrame(all_driver_coords).drop_duplicates(subset=["gene"])
all_driver_coords_df.to_csv(f"{DRIVER_GENES_DIR}/all_driver_genes_hg38.tsv", sep="\t", index=False)
