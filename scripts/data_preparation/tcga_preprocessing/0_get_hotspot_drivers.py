import pandas as pd

# input files/dirs
HOTSPOTS_RAW = "data/hotspots_v2"

# output files/dirs
HOTSPOTS = "data/hotspots_v2/hotspots_tcga.tsv"
HOTSPOTS_HG19 = "data/hotspots_v2/hotspots_tcga_hg19.bed"

indel_hotspots = pd.read_csv(f"{HOTSPOTS_RAW}/INDEL-hotspots.tsv", sep="\t")
snv_hotspots = pd.read_csv(f"{HOTSPOTS_RAW}/SNV-hotspots.tsv", sep="\t")
drivers_hotspots = pd.concat([indel_hotspots, snv_hotspots])
drivers_hotspots = drivers_hotspots[["Hugo_Symbol", "Genomic_Position", "Codon_Change", "Organ_Types", "qvalue"]]
drivers_hotspots = drivers_hotspots[drivers_hotspots["qvalue"] < 0.05]
drivers_hotspots["Genomic_Position"] = drivers_hotspots["Genomic_Position"].str.split("|")
drivers_hotspots = drivers_hotspots.explode("Genomic_Position")
drivers_hotspots["Genomic_Position"] = drivers_hotspots["Genomic_Position"].str.split("_").str[0]
drivers_hotspots = drivers_hotspots[["Hugo_Symbol", "Genomic_Position", "Organ_Types"]]
drivers_hotspots.drop_duplicates(inplace=True)
drivers_hotspots["Organ_Types"] = drivers_hotspots["Organ_Types"].str.split("|")
drivers_hotspots = drivers_hotspots.explode("Organ_Types")
drivers_hotspots["Organ_Types"] = drivers_hotspots["Organ_Types"].str.split(":").str[0]
drivers_hotspots.drop_duplicates(inplace=True)
drivers_hotspots.dropna(inplace=True)
print(drivers_hotspots.shape)
drivers_hotspots.to_csv(HOTSPOTS, sep="\t", index=False)

# create hg19 bed file for liftover

hotspot_bed = drivers_hotspots[["Genomic_Position"]].copy()
hotspot_bed["chrom"] = hotspot_bed["Genomic_Position"].apply(lambda x: x.split(":")[0])
hotspot_bed["start"] = hotspot_bed["Genomic_Position"].apply(lambda x: int(x.split(":")[1]) - 1)
hotspot_bed["end"] = hotspot_bed["Genomic_Position"].apply(lambda x: int(x.split(":")[1]))
hotspot_bed = hotspot_bed[["chrom", "start", "end"]]
hotspot_bed["chrom"] = "chr" + hotspot_bed["chrom"].astype(str)
hotspot_bed.drop_duplicates(inplace=True)
print(hotspot_bed.shape)
hotspot_bed.to_csv(HOTSPOTS_HG19, sep="\t", index=False, header=False)