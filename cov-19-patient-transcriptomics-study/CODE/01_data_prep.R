# one-time installs
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db"), ask = FALSE, update = FALSE)

# load required packages
library(readxl)            # read excel files
library(clusterProfiler)   # gene ontology & pathway databases
library(org.Hs.eg.db)      # human genome database

# set working directory
setwd("C:/Users/funny/OneDrive/Desktop/College/Grad School/ABT 785/COVID-19 Patient Transcriptomics/Project2")

##################
### patient data ###
##################

# Supplementary data 3 from publication
patient_data <- read_xlsx(
  "C:/Users/funny/OneDrive/Desktop/College/Grad School/ABT 785/COVID-19 Patient Transcriptomics/Project2/data/1-s2.0-S2001037020305419-mmc3.xlsx"
)

# make second row the column names
colnames(patient_data) <- patient_data[2, ]

# remove first two rows, and keep first four columns
patient_data <- patient_data[-c(1:2), 1:4]

# simplify column names
colnames(patient_data) <- c("Patient", "Age", "Sex", "Severity")

# simplify disease severity ranking
patient_data$Severity[grepl("Mild",     patient_data$Severity)] <- "mild"
patient_data$Severity[grepl("Severe",   patient_data$Severity)] <- "severe"
patient_data$Severity[grepl("Moderate", patient_data$Severity)] <- "moderate"

##################
### DEG analysis ###
##################

# results from differential gene expression analysis
dge_results <- read_xlsx(
  "C:/Users/funny/OneDrive/Desktop/College/Grad School/ABT 785/COVID-19 Patient Transcriptomics/Project2/data/1-s2.0-S2001037020305419-mmc2.xlsx",
  sheet = "Control_vs_COVID"
)

# remove weird empty row
dge_results <- subset(dge_results, is.na(dge_results$Gene) == FALSE)

##################
### human genes ###
##################

# attach database of human genes
data(geneList, package = "DOSE")

# load it
gene_db <- enrichGO(names(geneList), OrgDb = org.Hs.eg.db, readable = TRUE)

# convert to data frame
gene_db <- data.frame(Gene  = gene_db@gene2Symbol,
                      geneID = names(gene_db@gene2Symbol))

# subset to genes in our dataset
gene_db <- subset(gene_db, Gene %in% dge_results$Gene)

##################
### KEGG pathways ###
##################

# get database of human pathways & associated genes
pathway_db <- download_KEGG("hsa", keggType = "KEGG")

# extract pathway IDs and associated gene IDs
pathgenes_db <- pathway_db$KEGGPATHID2EXTID
colnames(pathgenes_db) <- c("hsaID", "geneID")

# extract pathway names and associated pathway IDs
pathway_db <- pathway_db$KEGGPATHID2NAME
colnames(pathway_db) <- c("hsaID", "Pathway")

# combine: pathway IDs + gene IDs + pathway IDs
pathways <- merge(pathgenes_db, gene_db, by = "geneID")

# combine: pathway IDs + gene IDs + pathway IDs + pathway names
pathways <- merge(pathways, pathway_db, by = "hsaID")

write.csv(dge_results, "R/DGE_results.csv",   row.names = FALSE, na = "", quote = FALSE)
write.csv(pathways,    "R/DGE_pathways.csv",  row.names = FALSE, na = "", quote = FALSE)

