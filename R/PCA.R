# Install and load PCA visualization utilities
install.packages("factoextra")
library(factoextra)
library(ggplot2)
cat("\014")
# ---------------------------------------------------------
# Loads and inspects the COMPAS dataset.
# Prepares demographic and criminal history variables from the COMPAS dataset
# Performs Principal Component Analysis (PCA) to explore structure in predictor variables
# Visualises how individuals distribute across PCs and whether structure aligns with race
# Identifies which variables contribute most to separation in the PCA space
# ---------------------------------------------------------


# Check working directory
getwd()

# Load dataset from CSV file
df <- read.csv("../Dataset/propublica_data_for_fairml.csv")


# Remove variables that should not be included in PCA:
# Remove target
# Remove score_factor as it is derived from COMPAS
df <- df[, !(names(df) %in% "Two_yr_Recidivism")]
df <- df[, !(names(df) %in% "score_factor")]


# Create a single categorical Race variable
# Any individual not matching the listed categories is assumed White.
df$Race <- ifelse(df$African_American == 1, "African American",
           ifelse(df$Asian == 1, "Asian",
           ifelse(df$Hispanic == 1, "Hispanic",
           ifelse(df$Native_American == 1, "Native American",
           ifelse(df$Other == 1, "Other", "White")))))


# Select relevant predictor variables for PCA.
X <- df[, c("Number_of_Priors",
            "Age_Above_FourtyFive",
            "Age_Below_TwentyFive",
            "African_American",
            "Asian",
            "Hispanic",
            "Native_American",
            "Other",
            "Female",
            "Misdemeanor")]



# ---------------------------------------------------------
# Run PCA
# ---------------------------------------------------------


# - center = TRUE standardizes mean to 0
# - scale = TRUE standardizes variance so variables contribute equally
pca_res <- prcomp(X, center = TRUE, scale. = TRUE)

# Summary shows proportion of variance explained by each principal component
summary(pca_res)

# ----------------------
# Plot 1
# Scree plot shows how much variance each principal component explains.
# ----------------------
plot(pca_res, type = "l") 

# Show importance of each variable in each PC
pca_res$rotation

# Store loadings
loadings <- pca_res$rotation
loadings

# Convert PCA results to a dataframe and attach Race for visualisation
scores <- as.data.frame(pca_res$x)   # PC scores
scores$Race <- df$Race


# ----------------------
# Plot 2
# Scatter plot of first two principal components (PC1 vs PC2)
# This helps identify whether PCA components encode racial differences.
# ----------------------
ggplot(scores, aes(x = PC1, y = PC2, colour = Race)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA of COMPAS Predictors (PC1 vs PC2)",
       x = "PC1", y = "PC2") +
  theme_minimal()


# ----------------------
# Plot 3 + 4
# Contribution plots showing which variables contribute most to PC1 and PC2.
# ----------------------
fviz_contrib(pca_res, choice = "var", axes = 1)  # PC1 contributions
fviz_contrib(pca_res, choice = "var", axes = 2)  # PC2 contributions

# Display ranked absolute contributions to PC1 and PC2 for interpretation
sort(abs(loadings[,1]), decreasing = TRUE)
sort(abs(loadings[,2]), decreasing = TRUE)


