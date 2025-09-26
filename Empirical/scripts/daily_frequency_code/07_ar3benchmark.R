suppressPackageStartupMessages({
  library(data.table)
  library(glmnet)
  library(doParallel)
  library(readr)  
  library(here)  
  library(ggplot2)
  library(dplyr)
  library(arrow)
  library(jsonlite)
})

# load data 
load(here("data", "X_data"))

# keep only return data
topic_cols <- grep("^Topic_", names(X_data), value = TRUE)
stock_cols <- setdiff(setdiff(names(X_data), "date"), topic_cols)
data <- X_data[, stock_cols]

# Function for AR(3) forecast 
roll_ar3 <- function(x, L = 30) {
  x <- as.numeric(x)
  n <- length(x)
  f <- rep(NA_real_, n)
  
  K <- 3L
  offset <- K - 1L         # drop first (K-1) to mirror Xlag's first_valid
  n2 <- n - offset         # length after the drop
  if (n2 <= L + 1L) return(f)
  
  for (t in (L + 1L):(n2 - 1L)) {
    idx <- (t - L):(t - 1L)
    yy <- x[offset + idx + 1L]
    XX <- cbind(1,
                x[offset + idx],
                x[offset + idx - 1L],
                x[offset + idx - 2L])
    ok_est  <- all(stats::complete.cases(cbind(yy, XX)))
    ok_pred <- all(stats::complete.cases(
      c(x[offset + t], x[offset + t - 1L], x[offset + t - 2L])
    ))
    if (ok_est && ok_pred) {
      fit  <- .lm.fit(XX, yy)
      beta <- fit$coefficients
      f[offset + t] <- c(1, x[offset + t], x[offset + t - 1L], x[offset + t - 2L]) %*% beta
    } else {
      f[offset + t] <- NA_real_
    }
  }
  f
}

# Apply function to each stock
forecasts_ar3 <- as_tibble(lapply(as.list(data), roll_ar3))
names(forecasts_ar3) <- names(data)

# Add the date that each row's forecast refers to (i.e., the t+1 date)
forecasts_ar3$date_pred <- dplyr::lead(X_data$date, 1L)

# --- save results ---
write_parquet(forecasts_ar3, file.path(here("output"), "forecasts_ar3.parquet"), compression = "zstd")
