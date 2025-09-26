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

# Load data 
load(here("data", "X_data"))
results <- list(
  forecasts    = as.data.table(read_parquet(file.path(here("output"), "forecasts.parquet"))),
  active_coefs = as.data.table(read_parquet(file.path(here("output"), "active_coefs.parquet"))),
  meta         = read_json(file.path(here("output"), "meta.json"), simplifyVector = TRUE)
)

# --- Data preparation  ---
X <- as.data.table(X_data)                
setorder(X, date)
permno_ticker <- colnames(X_data)
topic_cols <- grep("^Topic_", names(X), value = TRUE)
stock_cols <- setdiff(setdiff(names(X), "date"), topic_cols)
target_stocks <- as.character(permno_ticker[2:486])
K <- 3                                                  
L <- 30

# Build the same lagged features as in original Lasso
lag_block <- function(dt, cols, K, tag) {
  if (!length(cols)) return(NULL)
  do.call(cbind, lapply(0:(K-1), function(k) {
    out <- dt[, shift(.SD, n = k), .SDcols = cols]
    setnames(out, paste0(cols, "_", tag, "Lag", k))
    out
  }))
}

Xlag <- cbind(
  X[, .(date)],
  lag_block(X, stock_cols, K, "R"),
  lag_block(X, topic_cols, K, "T")
)

first_valid <- 1 + (K - 1)
Xlag <- Xlag[first_valid:.N]
pred_cols <- setdiff(names(Xlag), "date")
Xmat <- as.matrix(Xlag[, ..pred_cols])
dates_t <- Xlag$date
orig_idx <- first_valid:nrow(X)

coefs_dt <- copy(results$active_coefs)
setkey(coefs_dt, target_stock, date_t)

get_lasso_predictors <- function(ts, date_t_val, coefs_dt) {
  coefs_dt[target_stock == ts & date_t == date_t_val, predictor]
}

# 2) Precompute name->index map once
col_ix <- setNames(seq_len(ncol(Xmat)), colnames(Xmat))

roll_augmented_ar3 <- function(target_stock, coefs_dt) {
  n <- nrow(Xmat)
  f <- rep(NA_real_, nrow(X))
  if (n <= L + 1L) return(f)
  
  for (t in (L + 1L):(n - 1L)) {
    lo <- t - L; hi <- t - 1L
    X_win <- Xmat[lo:hi, , drop = FALSE]
    
    y_idx <- orig_idx[lo:hi] + 1L
    y_idx <- y_idx[y_idx <= nrow(X)]
    X_win <- X_win[seq_along(y_idx), , drop = FALSE]
    y_win <- X[[target_stock]][y_idx]
    
    current_date <- dates_t[t]
    lasso_predictors <- get_lasso_predictors(target_stock, current_date, coefs_dt)
    
    target_lag_cols <- paste0(target_stock, "_RLag", 0:2)
    all_predictors <- intersect(unique(c(target_lag_cols, lasso_predictors)), colnames(X_win))
    if (!length(all_predictors)) next
    
    pred_idx <- unname(col_ix[all_predictors]); pred_idx <- pred_idx[!is.na(pred_idx)]
    X_sel <- X_win[, pred_idx, drop = FALSE]
    
    keep <- is.finite(y_win) & complete.cases(X_sel)
    if (sum(keep) < max(5, ncol(X_sel) + 1L)) next
    
    fit <- tryCatch(.lm.fit(cbind(1, X_sel[keep, , drop = FALSE]), y_win[keep]), error = function(e) NULL)
    if (is.null(fit) || anyNA(fit$coefficients)) next
    
    pred_row <- Xmat[t, pred_idx, drop = FALSE]
    if (!all(complete.cases(pred_row))) next
    
    pred_idx_out <- orig_idx[t] + 1L
    if (pred_idx_out > nrow(X)) next
    
    beta0 <- fit$coefficients[1]
    betas <- fit$coefficients[-1]
    f[pred_idx_out] <- beta0 + drop(pred_row %*% betas)
  }
  f
}


# --- Apply augmented AR(3) to all target stocks ---
cat("Running augmented AR(3) with Lasso-selected variables...\n")

# Setup parallelization
n_cores <- parallel::detectCores(logical = TRUE)
cl <- makeCluster(max(1L, n_cores - 1L))
registerDoParallel(cl)

# Export necessary objects to cluster
clusterExport(cl, c("Xmat", "X", "dates_t", "orig_idx", "L", "K", "first_valid",
                    "get_lasso_predictors", "roll_augmented_ar3", "results"))

# Run in parallel
forecasts_augmented <- foreach(target = target_stocks, .combine = cbind,
                               .packages = c("data.table")) %dopar% {
                                 coefs_dt <- results$active_coefs
                                 roll_augmented_ar3(target, coefs_dt)
                               }

stopCluster(cl)

# Convert to data frame and add proper column names
forecasts_augmented <- as.data.frame(forecasts_augmented)
names(forecasts_augmented) <- target_stocks

# Add date column (prediction dates)
forecasts_augmented$date_pred <- X_data$date[2:nrow(X_data)]
if (nrow(forecasts_augmented) < (nrow(X_data) - 1)) {
  # Pad with NAs if needed
  n_missing <- (nrow(X_data) - 1) - nrow(forecasts_augmented)
  forecasts_augmented <- rbind(
    data.frame(matrix(NA, nrow = n_missing, ncol = ncol(forecasts_augmented)),
               stringsAsFactors = FALSE),
    forecasts_augmented
  )
  names(forecasts_augmented) <- c(target_stocks, "date_pred")
  forecasts_augmented$date_pred <- X_data$date[2:nrow(X_data)]
}

cat("Augmented AR(3) forecasting complete!\n")