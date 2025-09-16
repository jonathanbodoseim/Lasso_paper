# =============================================================================
# scripts/main.R — LASSO + plots + R2
# =============================================================================

suppressPackageStartupMessages({
  library(here)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(purrr)
})

# ---- Load functions ---------------------------------------------------
source(here::here("Empirical","scripts", "_load_functions.R"))

# ---- Configuration ------------------------------------------------------------
set.seed(123)

# Targets to forecast (character vector of column names in X_data)
# Extract id numbers from vector (optional)
targets <- permno_vec[1:20]

# LASSO settings
cfg <- list(
  week_col      = "week",
  topic_pref    = "Topic_",
  L             = 30,
  K_ret         = 3,
  K_topic       = 3,
  lambda_choice = "lambda.1se",
  nfolds_cv     = 10,
  parallel      = TRUE,
  verbose       = TRUE
)

# Output paths
path_out_dir        <- here::here("output")
path_plot_dir       <- here::here("output", "plots_active_predictors")
path_forecasts_csv  <- here::here("output", "lasso_forecasts.csv")
path_coefs_csv      <- here::here("output", "lasso_active_coefs.csv")
path_r2_csv         <- here::here("output", "r2_by_stock.csv")
dir.create(path_out_dir, showWarnings = FALSE, recursive = TRUE)


# ---- Run rolling LASSO --------------------------------------------------------
out <- run_lasso_ret_and_topics(
  X_data        = X_data,
  target_stocks = targets,
  week_col      = cfg$week_col,
  topic_pref    = cfg$topic_pref,
  L             = cfg$L,
  K_ret         = cfg$K_ret,
  K_topic       = cfg$K_topic,
  lambda_choice = cfg$lambda_choice,
  nfolds_cv     = cfg$nfolds_cv,
  parallel      = cfg$parallel,
  verbose       = cfg$verbose
)

# Save model outputs
if (nrow(out$forecasts))    readr::write_csv(out$forecasts,    path_forecasts_csv)
if (nrow(out$active_coefs)) readr::write_csv(out$active_coefs, path_coefs_csv)

# ---- Visualize active predictors ---------------------------------------------
# Set save_dir = NULL to only return ggplot objects without writing PNGs.
plots <- plot_active_predictors_per_target(
  out            = out,
  permno_ticker  = permno_ticker,
  top_n          = 80,
  save_dir       = NULL # or path_plot_dir to write PNGs
)

# Example: print one plot (guarded)
# print(plots[[1]])

# ---- Diagnostics: mean number of active regressors per week -------------------
mean_active_by_week <- out$forecasts %>%
  as_tibble() %>%
  group_by(week_t) %>%
  summarise(mean_active = mean(n_active, na.rm = TRUE), .groups = "drop")

print(mean_active_by_week)

# ---- Match forecasts with actual returns -------------------------------------
# 1) extract forecasts
fc <- out$forecasts %>%
  as_tibble() %>%
  select(target_stock, week_pred, forecast)

# 2) extract actual returns
act <- weekly_wide %>%
  as_tibble() %>%
  pivot_longer(
    cols = -all_of(cfg$week_col),
    names_to = "target_stock",
    values_to = "actual"
  ) %>%
  rename(week = !!cfg$week_col)

# 3) join on stock id + week 
df_match <- fc %>%
  left_join(act, by = c("target_stock" = "target_stock", "week_pred" = "week"))

# ---- Compute R2 per stock --------------------------------
r2 <- function(f, a) {
  ss_res <- sum((a - f)^2)
  ss_tot <- sum((a - mean(a))^2)
  tibble(r2 = 1 - ss_res / ss_tot, n_obs = length(a))
}

r2_by_stock <- df_match %>%
  group_by(target_stock) %>%
  summarise(r2(forecast, actual), .groups = "drop") %>%
  arrange(desc(r2))

print(r2_by_stock)
readr::write_csv(r2_by_stock, path_r2_csv)

# ---- Plot Lambda vs. Active Predictors --------------------------------

plot_lambda_vs_predictors(out)




