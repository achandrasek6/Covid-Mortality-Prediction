# load required packages
library(ggplot2)
library(ggpubr)

# set working directory
setwd("C:/Users/funny/OneDrive/Desktop/College/Grad School/ABT 785/COVID-19 Patient Transcriptomics/Project2")

##################
### data prep ###
##################

# load data from previous script
dge_results <- read.csv("R/DGE_results.csv",  stringsAsFactors = FALSE)
pathways    <- read.csv("R/DGE_pathways.csv", stringsAsFactors = FALSE)

# pathways shown in plot
up_paths <- c(
  "Cytokine-cytokine receptor interaction",
  "JAK-STAT signaling pathway",
  "Complement and coagulation cascades",
  "Hematopoietic cell lineage",
  "Chemokine signaling pathway",
  "Inflammatory bowel disease",
  "Toll-like receptor signaling pathway",
  "IL-17 signaling pathway",
  "Th1 and Th2 cell differentiation",
  "TGF-beta signaling pathway"
)

down_paths <- c(
  "Ribosome",
  "Oxidative phosphorylation",
  "Protein processing in endoplasmic reticulum",
  "Phagosome",
  "Ferroptosis",
  "Type I diabetes mellitus",
  "Allograft rejection",
  "Viral myocarditis",
  "Oxytocin signaling pathway",
  "Amyotrophic lateral sclerosis"
)

# subset by those pathways
pathways <- subset(pathways, Pathway %in% c(up_paths, down_paths))

# combine with DGE results (common column = Gene)
pathways <- merge(pathways, dge_results, by = "Gene")

##################
### average p values ###
##################

# create empty columns for number of genes per pathway ...
pathways$NumGenes <- NA

# ... and the mean adjusted p value per pathway
pathways$pMean <- NA

for (path in c(up_paths, down_paths)) {                # loop through pathway names
  df <- subset(pathways, Pathway == path)              # subset pathway data by pathway name
  n  <- length(df$Gene)                                # get number of genes in that pathway
  p  <- mean(df$padj, na.rm = TRUE)                    # calculate mean p values of those genes
  pathways$NumGenes[pathways$Pathway == path] <- n     # add number of genes to column
  pathways$pMean[pathways$Pathway == path]  <- p       # add mean p value to column
}

# subset to up/down regulated pathways
pathways <- unique(pathways[c("Pathway", "NumGenes", "pMean")])
up_pathways   <- subset(pathways, Pathway %in% up_paths)
down_pathways <- subset(pathways, Pathway %in% down_paths)

##################
### order axis ###
##################

# y axis is ordered by number of genes
up_order <- up_pathways$Pathway[order(up_pathways$NumGenes, decreasing = TRUE)]

### BASE
ggplot(up_pathways,
       aes(x = NumGenes, y = Pathway)) +
  
  ### PLOT
  geom_point(aes(color = pMean,
                 size  = NumGenes)) +
  scale_y_discrete(limits = up_order)   # order y axis

##################
### customize ###
##################

### BASE
up.plot <-
  ggplot(up_pathways,
         aes(x = NumGenes, y = Pathway)) +
  
  ### PLOT
  geom_point(aes(color = pMean, size = NumGenes)) +
  
  ### MODS
  scale_color_gradient(low = "blue", high = "red",
                       limits = c(min(pathways$pMean, na.rm = TRUE),
                                  max(pathways$pMean, na.rm = TRUE))) +   # shared scale
  scale_size_continuous(limits = c(min(pathways$NumGenes, na.rm = TRUE),
                                   max(pathways$NumGenes, na.rm = TRUE))) + # shared scale
  theme_light() +
  theme(axis.title = element_blank()) +
  labs(color = "Mean\nadjusted\np-value",
       size  = "Number\nof genes")

##################
### down plot ###
##################

# order y axis by number of genes (down-regulated)
down_order <- down_pathways$Pathway[order(down_pathways$NumGenes, decreasing = TRUE)]

down.plot <-
  ggplot(down_pathways, aes(x = NumGenes, y = Pathway)) +
  geom_point(aes(color = pMean, size = NumGenes)) +
  scale_color_gradient(low = "blue", high = "red",
                       limits = c(min(pathways$pMean, na.rm = TRUE),
                                  max(pathways$pMean, na.rm = TRUE))) +
  scale_size_continuous(limits = c(min(pathways$NumGenes, na.rm = TRUE),
                                   max(pathways$NumGenes, na.rm = TRUE))) +
  scale_x_reverse() +
  scale_y_discrete(limits = down_order, position = "right") +
  theme_light() +
  theme(axis.title = element_blank()) +
  labs(color = "Mean\nadjusted\np-value",
       size  = "Number\nof genes")

# ensure up.plot has ordered y as well
up_order <- up_pathways$Pathway[order(up_pathways$NumGenes, decreasing = TRUE)]
up.plot <-
  ggplot(up_pathways, aes(x = NumGenes, y = Pathway)) +
  geom_point(aes(color = pMean, size = NumGenes)) +
  scale_color_gradient(low = "blue", high = "red",
                       limits = c(min(pathways$pMean, na.rm = TRUE),
                                  max(pathways$pMean, na.rm = TRUE))) +
  scale_size_continuous(limits = c(min(pathways$NumGenes, na.rm = TRUE),
                                   max(pathways$NumGenes, na.rm = TRUE))) +
  scale_y_discrete(limits = up_order) +
  theme_light() +
  theme(axis.title = element_blank()) +
  labs(color = "Mean\nadjusted\np-value",
       size  = "Number\nof genes")

# combine (ggpubr, stacked, shared legend on right, white background)
library(ggpubr)

dot.plot <- ggarrange(
  plotlist = list(up.plot, down.plot),   # <- robust combine
  ncol = 1,
  heights = c(1, 1),
  common.legend = TRUE,
  legend = "right",
  align = "v"
)

# show and save
dot.plot
dir.create("PLOTS", showWarnings = FALSE, recursive = TRUE)
ggsave("PLOTS/dot_plot.pdf", dot.plot, device = "pdf",
       width = 8, height = 8, units = "in")
