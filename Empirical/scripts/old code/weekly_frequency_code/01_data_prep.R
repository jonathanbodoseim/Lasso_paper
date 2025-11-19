# =============================================================================
# Project: Lasso Paper — Data Prep & Diagnostics
# Author:  (your name)
# Date:    2025-09-12
# Notes:
# - Uses relative paths with `here::here()` (no setwd)
# - Tidyverse style: snake_case, pipes, left_join, pivot_wider
# - Safer reads for .rds via readRDS()
# - Includes basic input checks and robust weekly compounding
# =============================================================================

# ---- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readxl)
  library(readr)     # for write_csv()
  library(lubridate)
  library(purrr)
  library(tseries)   # adf.test
  library(here)      # project-rooted paths
})

# ---- Paths -------------------------------------------------------------------
# Adjust these relative paths to match your repo layout
path_returns_xlsx <- here("data", "return_data.xlsx")
path_model_rds    <- here("data", "model87_f.rds")
path_topics_rds   <- here("data", "topic_names.rds")
path_articles_rds <- here("data", "clean_articles.rds")
path_out_X_csv    <- here("output", "X_data.csv")
path_out_permno_csv    <- here("output", "permno.csv")

# ---- Constants ---------------------------------------------------------------
bad_permnos <- c(
  "11703","16087","16309","16431","16581","16692","16736","16851","17307",
  "17700","17942","18143","18224","18267","18312","18420","18421","18428",
  "18576","18592","18724","18726","18911","19285","19286","20626","42534",
  "51692","59192","61209","84342","87404","90441","90442", "91907"
)

# ---- Helpers -----------------------------------------------------------------
safe_weekly_return <- function(ret_vec) {
  # Returns NA if all returns are NA; else compound (1 + r) - 1
  if (all(is.na(ret_vec))) return(NA_real_)
  prod(1 + ret_vec[!is.na(ret_vec)]) - 1
}

# ---- Load & Clean Return Data ------------------------------------------------
returns_raw <- read_excel(path_returns_xlsx)

returns_clean <- returns_raw %>%
  mutate(
    date = as.Date(date),
    RET  = suppressWarnings(as.numeric(RET))
  ) %>%
  filter(!PERMNO %in% bad_permnos) %>%
  mutate(week = floor_date(date, unit = "week", week_start = 1))

# Keep PERMNO–TICKER map if needed later
permno_ticker <- returns_clean %>%
  distinct(PERMNO, TICKER) %>%
  arrange(PERMNO)

permno_vec <- permno_ticker$PERMNO

# Weekly compounded returns per PERMNO
weekly_returns <- returns_clean %>%
  group_by(PERMNO, week) %>%
  summarise(weekly_return = safe_weekly_return(RET), .groups = "drop")

# Wide matrix: rows = weeks, columns = PERMNOs
weekly_wide <- weekly_returns %>%
  select(week, PERMNO, weekly_return) %>%
  pivot_wider(
    names_from  = PERMNO,
    values_from = weekly_return,
    values_fill = NA_real_
  ) %>%
  arrange(week)

# ---- Load Topic Model & Articles ---------------------------------------------
# .rds files should be read with readRDS(); they return an object directly.
load(path_model_rds)
load(path_topics_rds)     # kept in case you need labels
load(path_articles_rds)  # expected to contain a `date` column

# ---- Topic Probabilities per Article -----------------------------------------
# Assumes model87_f$theta is a document-topic matrix (rows = docs, cols = topics)
topic_probs <- as.data.frame(model87_f$theta)
colnames(topic_probs) <- paste0("Topic_", seq_len(ncol(topic_probs)))

# Bind topic probabilities to articles (remove any pre-existing Topic_ cols)
articles_topics <- df %>%
  select(-starts_with("Topic_"), everything()) %>%
  bind_cols(topic_probs)

# ---- Weekly Topic Shares ------------------------------------------------------
weekly_topics <- articles_topics %>%
  mutate(
    date = as.Date(date),
    week = floor_date(date, unit = "week", week_start = 1)
  ) %>%
  group_by(week) %>%
  summarise(
    across(starts_with("Topic_"), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  arrange(week)

# ---- Merge Weekly Returns & Weekly Topics ------------------------------------
X_data <- weekly_wide %>%
  left_join(weekly_topics, by = "week")

# Ensure output directory exists and write CSV
dir.create(dirname(path_out_X_csv), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(path_out_permno_csv), showWarnings = FALSE, recursive = TRUE)

write_csv(X_data, path_out_X_csv)
write_csv(permno_ticker, path_out_permno_csv)
