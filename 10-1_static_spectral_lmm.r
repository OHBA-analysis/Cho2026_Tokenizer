# ---------- R SETUP ----------
# Install once if needed:
# install.packages(c("lme4", "lmerTest", "emmeans", "readr", "dplyr"))

options(contrasts = c("contr.sum","contr.poly"))
# for Type III ANOVA with sum-to-zero contrasts (effects relative to grand mean)

# Load libraries
library(lme4)
library(lmerTest)
library(emmeans)
library(readr)
library(dplyr)

# Set user hyperparameters
metric_name = "l2"
metric_file_path = "data/static_gt1_l2.csv"

# ---------- Load data ----------
df <- read_csv(metric_file_path, show_col_types = FALSE) |>
  mutate(
    subject = factor(subject),
    model   = factor(model),
    dataset = factor(dataset),
    channel = factor(channel),
  )

# Save per-subject mean across datasets and channels (one point per subject per model)
subj_means <- df |>
  group_by(subject, model) |>
  summarise(mean_metric = mean(metric, na.rm = TRUE), .groups = "drop")
write_csv(subj_means, paste0("subject_means_", metric_name, ".csv"))

# ---------- LMM: dataset as fixed; channel as random ----------
# Use ML (REML = FALSE) for fixed-structure comparison
# (i.e., when doing ANOVAs/LRTs between models with different fixed structures)
# ML = maximum likelihood; REML = restricted maximum likelihood
# lmer() fits a linear mixed-effects model (LMM)

# Random intercept for subject and channel; fixed model, dataset, and their interaction
lmm_full_ML <- lmer(metric ~ model * dataset + (1 | subject) + (1 | channel), data = df, REML = FALSE)
lmm_reduced_ML <- lmer(metric ~ model + dataset + (1 | subject) + (1 | channel), data = df, REML = FALSE)  # no interaction

# Type III tests (Satterthwaite df) for fixed effects, incl. model:dataset interaction
anova_full_ML <- anova(lmm_full_ML, ddf = "Satterthwaite")
anova_reduced_ML <- anova(lmm_reduced_ML, ddf = "Satterthwaite")
print(anova_full_ML)
print(anova_reduced_ML)

# Optional likelihood-ratio test (LRT) against reduced model
lrt <- anova(lmm_reduced_ML, lmm_full_ML)
print(lrt)

# ---------- Final model for estimation (REML) ----------
# Select final model based on significance of interaction
lmm <- lmer(metric ~ model * dataset + (1 | subject) + (1 | channel), data = df, REML = TRUE)
anova_lmm <- anova(lmm, ddf = "Satterthwaite")
print(anova_lmm)

# NOTE: Based on the comparison of models with and without interaction,
#       the interaction was not significant. We would typically select the simpler 
#       model without interaction. However, as we want to test for the interaction
#       model × dataset (to show consistency across datasets), we keep the
#       interaction in the final model.

# Visual inspection of ANOVA assumptions
plot(lmm) # residuals vs fitted; linearity, homoscedasticity
qqnorm(resid(lmm)); qqline(resid(lmm))  # residual normality (slight tail deviations are expected with large N)
qqnorm(ranef(lmm)$subject[[1]]); qqline(ranef(lmm)$subject[[1]])  # normality of random effects

# Check for singularity
sgl_check <- isSingular(lmm, tol=1e-4)  # should be FALSE
print(paste("Is the model singular?", sgl_check))
# NOTE
# - isSingular() tests whether any random-effect variance estimates are (numerically) ~= 0,
#   or whether random effects are perfectly correlated. In these cases the random-effects
#   variance-covariance matrix is "singular."
# - A singular fit means the model is estimating unnecessary or redundant random effects
#   (e.g., a random intercept or slope that contributes no variability).
# - Use VarCorr(lmm) to inspect which variance component is ~= 0.
# - If singularity occurs, consider simplifying the random-effects structure by
#   removing the component(s) with ~0 variance. This avoids overfitting and improves
#   model stability.

# ---------- EMMs ----------
# 1) EMM per model -> for reporting & violin overlay
emm_model <- emmeans(lmm, ~ model)  # averages over datasets, channels, and subjects
emm_model_df <- as.data.frame(emm_model)
write_csv(emm_model_df, paste0("emm_model_", metric_name, ".csv"))

# Pairwise comparisons among models with multiplicity correction
pairs_model <- pairs(emm_model, adjust = "tukey")  # or "holm"
pairs_model_df <- as.data.frame(summary(pairs_model))
write_csv(pairs_model_df, paste0("pairwise_model_contrasts_", metric_name, ".csv"))

# 2) EMMs per model × dataset -> for interaction plot (one line per dataset)
emm_md <- emmeans(lmm, ~ model | dataset)  # EMMs of model within each dataset (averaged over channels and subjects)
emm_md_df <- as.data.frame(emm_md)
write_csv(emm_md_df, paste0("emm_model_by_dataset_", metric_name, ".csv"))

# Save the specific line for the model:dataset interaction test (from Type III table)
int_row <- as.data.frame(anova_lmm)
int_row$term <- rownames(anova_lmm)
int_row <- int_row[grep("model:dataset", int_row$term), , drop = FALSE]
write_csv(int_row, paste0("model_dataset_interaction_test_", metric_name, ".csv"))
