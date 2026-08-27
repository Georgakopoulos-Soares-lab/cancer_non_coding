import os
import argparse
import pandas as pd

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mut_file", type=str, help="Path to the mutation file")
	parser.add_argument("--annovar_dir", type=str, default="", help="Path to the annovar directory")
	parser.add_argument("--results", type=str, default="", help="Directory to save annotated mutation file")
	args = parser.parse_args()
	mut_file = args.mut_file
	annovar_dir = args.annovar_dir
	results = args.results

	if len(results) > 0 and not os.path.exists(results):
		os.makedirs(results)

	if len(annovar_dir) == 0:
		annovar_dir = "annovar"

	mut_df = pd.read_csv(mut_file, sep="\t")
	print(f"Read {len(mut_df)} mutations from {mut_file}")
	filename = os.path.basename(mut_file).replace('.tsv', '')

	if mut_df.shape[0] == 0:
		print("No mutations found. Exiting.")
	else:
		# generate input file for annovar
		mut_bed = mut_df[["mutation"]].copy()
		mut_bed["chrom"] = mut_bed["mutation"].apply(lambda x: x.split(":")[0])
		mut_bed["start"] = mut_bed["mutation"].apply(lambda x: int(x.split(":")[1].split("-")[0]) - 1)  # annovar uses 0-based start
		mut_bed["end"] = mut_bed["mutation"].apply(lambda x: int(x.split(":")[1].split("-")[1]) - 1)
		mut_bed["ref"] = mut_bed["mutation"].apply(lambda x: x.split(":")[2])
		mut_bed["alt"] = mut_bed["mutation"].apply(lambda x: x.split(":")[3])
		mut_bed["end"] = mut_bed.apply(lambda x: x["start"] if x["ref"] == "-" else x["end"], axis=1) # handle insertions for annovar
		mut_bed = mut_bed[["chrom", "start", "end", "ref", "alt", "mutation"]]
		mut_bed.drop_duplicates(inplace=True)
		mut_bed.to_csv(f"{filename}.input", sep="\t", index=False, header=False)

		# download annovar database if not already present
		if not os.path.exists(f"{annovar_dir}/humandb/"):
			os.makedirs(f"{annovar_dir}/humandb/")
			cmd = f"{annovar_dir}/annotate_variation.pl -buildver hg38 -downdb -webfrom annovar refGene {annovar_dir}/humandb/"
			os.system(cmd)

		# annotate mutations with genic regions using annovar
		input_file = f"{filename}.input"
		output_prefix = f"{filename}"
		genome_db = f"{annovar_dir}/humandb/"
		cmd = f"{annovar_dir}/annotate_variation.pl -out {output_prefix} -build hg38 {input_file} {genome_db}"
		os.system(cmd)

		# add genic region annotations from annovar to the mutation data
		annot_df = pd.read_csv(f"{filename}.variant_function", sep="\t", header=None)
		annot_df.columns = ["genic_region", "gene", "chr", "start", "end", "ref", "alt", "mutation"]
		annot_df = annot_df[["mutation", "genic_region", "gene"]]
		annot_df.drop_duplicates(inplace=True)
		mut_df = mut_df.merge(annot_df, on="mutation", how="left")
		print(f"Annotated {mut_df.shape[0]} mutations with gene and genic regions")

		# add functional effect for exonic variants
		exonic_file = f"{filename}.exonic_variant_function"
		if os.path.exists(exonic_file) and os.path.getsize(exonic_file) > 0:
			exonic_annot_df = pd.read_csv(exonic_file, sep="\t", header=None)
			exonic_annot_df.columns = ["line", "func_effect", "var", "chr", "start", "end", "ref", "alt", "mutation"]
			exonic_annot_df = exonic_annot_df[["mutation", "func_effect"]]
			exonic_annot_df.drop_duplicates(inplace=True)
			mut_df = mut_df.merge(exonic_annot_df, on="mutation", how="left")
			print(f"Annotated {mut_df.shape[0]} mutations with functional effects")
		else:
			print("No exonic variants found.")

	if len(results) > 0:
		results = f"{results}/{filename}_annot.tsv"
	else:
		results = f"{filename}_annot.tsv"
	mut_df.to_csv(results, sep="\t", index=False)
