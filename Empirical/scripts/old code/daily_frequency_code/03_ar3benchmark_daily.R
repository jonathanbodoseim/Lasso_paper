# Setup
suppressPackageStartupMessages({
  library(data.table)
  library(here)
  library(dplyr)
  library(tibble)
  library(ggplot2)
  library(arrow)
  library(scales)
})

# Load original data and convert immediately to data.table (for efficiency)
load(here("data", "X_data"))
X <- as.data.table(X_data)
if (!inherits(X$date, "POSIXt")) X[, date := as.POSIXct(date, tz = "UTC")]

# Determine column sets once
topic_cols <- grep("^Topic_", names(X), value = TRUE)
stock_cols <- setdiff(names(X), c("date", topic_cols))

# Prepare data for the AR(3) function (as a tibble/data.frame for dplyr/across)
data_returns <- X %>% select(all_of(stock_cols)) %>% as_tibble()

roll_ar3_forecast <- function(x, L = 30) {
  x <- as.numeric(x); n <- length(x); f <- rep(NA_real_, n)
  if (anyNA(x)) {
    idx <- which(!is.na(x))
    x <- approx(x = idx, y = x[idx], xout = seq_len(n), method = "linear", rule = 2)$y
  }
  for (t in (L + 3):(n - 1)) {
    s_lo <- t - L; s_hi <- t - 1
    yy <- x[(s_lo + 1):(s_hi + 1)]
    XX <- cbind(1, x[s_lo:s_hi], x[(s_lo - 1):(s_hi - 1)], x[(s_lo - 2):(s_hi - 2)])
    b  <- .lm.fit(XX, yy)$coefficients
    f[t + 1] <- c(1, x[t], x[t - 1], x[t - 2]) %*% b
  }
  f
}

# --- forecasts already as tibble ---
forecasts_ar3 <- data_returns %>%
  mutate(across(everything(), roll_ar3_forecast)) %>%
  rename_with(~ sub("^forecast_", "", paste0("forecast_", .x))) %>%
  add_column(date_t = X$date, .before = 1) %>%
  as_tibble()

##################################################################################
write_parquet(forecasts_ar3, file.path(here("output"), "forecasts_ar3.parquet"))
##################################################################################