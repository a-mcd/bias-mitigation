install.packages("ggplot2")
install.packages("dplyr")
library(ggplot2)
library(dplyr)

cat("\014")
# ---------------------------------------------------------
# Loads and inspects the COMPAS dataset.
# Creates categorical demographic variables (Race, Sex).

# Produces 4 plots exploring potential disparities:
### Representation by race and gender
### Prior convictions distribution across racial groups
### Recidivism outcomes split by race and sex
### Recidivism outcomes by age
# ---------------------------------------------------------

# Check working directory
getwd()

# Load dataset from CSV file
df <- read.csv("../Dataset/propublica_data_for_fairml.csv")

summary(df) # View summary statistics for all columns
colSums(is.na(df)) # Check for missing values in each column


# Create a single categorical Race variable
# Any individual not matching the listed categories is assumed White.
df$Race <- ifelse(df$African_American == 1, "African American",
           ifelse(df$Asian == 1, "Asian",
           ifelse(df$Hispanic == 1, "Hispanic",
           ifelse(df$Native_American == 1, "Native American",
           ifelse(df$Other == 1, "Other", "White")))))

# Convert Race to a factor for plotting and analysis
df$Race <- factor(df$Race)

# Create a categorical Sex variable from binary Female indicator
df$Sex <- ifelse(df$Female == 1, "Female", "Male")
df$Sex <- factor(df$Sex)

# Convert recidivism label (0/1) into readable factor labels
df$Two_yr_Recidivism <- factor(df$Two_yr_Recidivism,
                               labels = c("No Recidivism", "Recidivism"))


# Create an age group variable from dummy columns
# Age_Below_TwentyFive = 1 -> "Below 25"
# Age_Above_FourtyFive = 1 -> "Above 45"
# Otherwise                 -> "25–45"
df <- df %>%
  mutate(
    Age_Group = case_when(
      Age_Below_TwentyFive == 1 ~ "Below 25",
      Age_Above_FourtyFive == 1 ~ "Above 45",
      TRUE                      ~ "25–45"
    ),
    Age_Group = factor(Age_Group, levels = c("Below 25", "25–45", "Above 45"))
  )

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------


# ----------------------
# Plot 1
# Bar chart showing number of individuals by Race and Sex
# Identifies demographic representation in the dataset.
# ----------------------
ggplot(df, aes(x = Race, fill = Sex)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Count of Individuals by Race and Gender",
    x = "Race",
    y = "Count",
    fill = "Sex"
  ) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))


# ----------------------
# Plot 2
# Density distribution of Number of Prior Convictions
# Same scale is used so charts can be compared
# ----------------------
ggplot(df, aes(x = Number_of_Priors, fill = Race)) +
  geom_density(alpha = .6) +
  facet_wrap(~ Race) +
  labs(
    title = "Distribution of Prior Convictions by Race",
    x = "Number of Prior Convictions",
    y = "Density (Proportion of the dataset in that range)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")



# ----------------------
# Plot 3
# Stacked bar chart of recidivism outcome
# Split by gender and faceted by race.
# ----------------------
ggplot(df, aes(x = Sex, fill = Two_yr_Recidivism)) +
  geom_bar(position = "fill") +
  facet_wrap(~ Race) +
  labs(
    title = "Recidivism Outcome by Race and Sex",
    x = "Sex",
    y = "Proportion",
    fill = "Outcome"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()



# ----------------------
# Plot 4
# Distribution of recidivism by age group
# ----------------------
ggplot(df, aes(x = Age_Group, fill = Two_yr_Recidivism)) +
  geom_bar(position = "fill") +
  labs(
    title = "Recidivism Proportion by Age Group",
    x = "Age Group",
    y = "Proportion",
    fill = "Outcome"
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()




