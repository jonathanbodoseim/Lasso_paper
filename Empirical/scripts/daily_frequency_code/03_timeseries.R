library(dplyr)
library(forecast)
library(tseries)
library(purrr)
library(tibble)

# load data
load(here("output", "model20"))
load(here("data", "clean_articles.rds"))

# Assign each article its topic probabilities from the stm
topic_probs <- as.data.frame(model20$theta)
df <- bind_cols(df, topic_probs)
topic_share_daily <- df %>%
  group_by(date) %>%
  summarise(across(starts_with("V"), mean, na.rm = TRUE))

extract_innovations_safe <- function(series) {
  n <- length(series)
  if (n == 0 || all(is.na(series))) return(rep(NA_real_, n))
  
  # work on the longest contiguous non-NA block to preserve alignment
  idx <- which(!is.na(series))
  if (length(idx) < 4) return(rep(NA_real_, n))
  runs <- split(idx, cumsum(c(1, diff(idx) != 1)))
  block <- runs[[ which.max(lengths(runs)) ]]
  y <- series[block]
  if (length(y) < 4) return(rep(NA_real_, n))
  
  # stationarity check + (up to) 2 differences
  p_value <- suppressWarnings(tryCatch(tseries::adf.test(y, alternative = "stationary")$p.value,
                                       error = function(e) NA_real_))
  d <- 0; max_d <- 2
  while (!is.na(p_value) && p_value > 0.05 && d < max_d && length(y) > 3) {
    y <- diff(y); d <- d + 1
    p_value <- suppressWarnings(tryCatch(tseries::adf.test(y, alternative = "stationary")$p.value,
                                         error = function(e) NA_real_))
  }
  
  fit <- suppressWarnings(tryCatch(
    forecast::Arima(y, order = c(1, d, 0), include.mean = TRUE),
    error = function(e) NULL
  ))
  
  out <- rep(NA_real_, n)
  if (!is.null(fit)) {
    res <- as.numeric(residuals(fit))   # length(res) == length(y)
    out[block] <- res                   # align back into original positions
  }
  out
}

# Wrapper: apply to all topic columns (default: V1…V20) and keep date
extract_innovations_topics <- function(df, date_col = "date", topic_cols = NULL, prefix = "innov_") {
  if (is.null(topic_cols)) {
    topic_cols <- grep("^V\\d+$", names(df), value = TRUE)
  }
  innov_list <- lapply(df[topic_cols], extract_innovations_safe)
  innov_df <- as.data.frame(innov_list, optional = TRUE)
  names(innov_df) <- paste0(prefix, topic_cols)
  cbind(df[date_col], innov_df)
}

topic_innov_daily <- extract_innovations_topics(df)

names(topic_innov_daily) <- sub("^innov[_]?V?(\\d+)$", "Topic_\\1", names(topic_innov_daily))



################ save data #############################
save(topic_innov_daily, file = here("data", "topic_innov_daily"))

