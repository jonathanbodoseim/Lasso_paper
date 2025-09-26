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
path_topic_innovations <- here("data", "topic_innov_daily")

# ---- Load Data ------------------------------------------------
returns <- read_excel(path_returns_xlsx)
load(path_topic_innovations)


# ---- Constants ---------------------------------------------------------------
bad_permnos <- c(
  "11703","16087","16309","16431","16581","16692","16736","16851","17307",
  "17700","17942","18143","18224","18267","18312","18420","18421","18428",
  "18576","18592","18724","18726","18911","19285","19286","20626","42534",
  "51692","59192","61209","84342","87404","90441","90442", "91907"
)

# ---- Clean Data ------------------------------------------------
returns <- returns %>%
  mutate(
    date = as.Date(date),
    RET  = suppressWarnings(as.numeric(RET))
  )

returns <- returns[]

topic_innov_daily <- topic_innov_daily %>%
  mutate(date = as.Date(date))

topics_wide <- topic_innov_daily %>%
  group_by(date) %>%
  summarise(across(starts_with("Topic_"), ~mean(.x, na.rm = TRUE)), .groups = "drop")

returns_wide <- returns %>%
  select(PERMNO, date, RET) %>%  # keep only what you need
  pivot_wider(
    names_from = PERMNO,
    values_from = RET
  )

returns_wide <- returns_wide[, !(names(returns_wide) %in% bad_permnos)]

X_data <- returns_wide %>%
  left_join(topics_wide, by = "date")


# ---- Save Data ------------------------------------------------
save(X_data, file = here("data", "X_data"))




