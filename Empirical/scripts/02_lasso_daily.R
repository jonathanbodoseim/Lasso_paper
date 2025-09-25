suppressPackageStartupMessages({
  library(data.table)
  library(glmnet)
  library(doParallel)
  library(readr)  
  library(here)   
  library(ggplot2)
})

load(here("data", "X_data"))

# --- inputs ---
X <- as.data.table(X_data)                
setorder(X, date)
topic_cols <- grep("^Topic_", names(X), value = TRUE)
stock_cols <- setdiff(setdiff(names(X), "date"), topic_cols)
target_stocks <- stock_cols
K <- 3                                                    
L <- 30 
gap <- K          # use only lags outside of training window for forecast
nfolds <- 10 

set.seed(1)

# --- build 3 lags for returns and topics ---
lag_block <- function(dt, cols, K, tag) {
    lags <- lapply(0:(K-1), function(k) {
    lagged <- dt[, shift(.SD, n = k), .SDcols = cols]
    setnames(lagged, paste0(cols, "_", tag, "Lag", k))
    lagged
  })
  do.call(cbind, lags)
}

Xlag <- cbind(
  X[, .(date)],
  lag_block(X, stock_cols, K, "R"),
  lag_block(X, topic_cols, K, "T")
)

first_valid <- 1 + (K - 1)                 
Xlag <- Xlag[first_valid:.N]
pred_cols <- setdiff(names(Xlag), "date")  
Xmat     <- as.matrix(Xlag[, ..pred_cols])
dates_t  <- Xlag$date
orig_idx <- first_valid:nrow(X)           

# use all available cores
n_cores <- parallel::detectCores(logical = TRUE)
cl <- makeCluster(max(1L, n_cores - 1L))   # leave 1 core free
registerDoParallel(cl)

# setup lists to store results
forecasts_4 <- list()
coefs_4     <- list()

# setup progress bar
total_iterations <- 0 
for (target in target_stocks) {
  max_t <- nrow(Xmat) - 1L
  if (max_t >= L) {
    total_iterations <- total_iterations + (max_t - L)
  }
}
current_iteration <- 0
cat("Running rolling lasso for", length(target_stocks), "stocks...\n")
cat("Progress: [", rep(" ", 50), "]\r", sep = "")


# --- rolling lasso per target ---

for (target in target_stocks) {
  max_t <- nrow(Xmat) - 1L
  if (max_t < (L + gap)) next
  
  for (t in (L + gap):max_t) {

    hi <- t - gap
    lo <- hi - (L - 1L)
    X_win <- Xmat[lo:hi, , drop = FALSE]
    y_idx <- orig_idx[lo:hi] + 1L
    y_idx <- y_idx[y_idx <= nrow(X)]
    X_win <- X_win[seq_along(y_idx), , drop = FALSE]
    y_win <- X[[target]][y_idx]
    
    keep <- complete.cases(X_win) & is.finite(y_win)
    if (sum(keep) >= 5) {
      fit <- cv.glmnet(
        x = X_win[keep, , drop = FALSE],
        y = y_win[keep],
        alpha = 1, family = "gaussian", nfolds = nfolds,
        standardize = TRUE, parallel = TRUE
      )
      
      lambda_min  <- fit$lambda.min
      lambda_max  <- fit$lambda[1]
      lambda_norm <- lambda_min / lambda_max
      
      f <- as.numeric(predict(fit, newx = Xmat[t, , drop = FALSE], s = "lambda.min"))
      
      b  <- as.matrix(coef(fit, s = "lambda.min"))[-1, , drop = TRUE]
      n_predictors <- sum(b != 0)
      
      wp_idx <- orig_idx[t] + 1L
      date_pred_val <- if (wp_idx <= nrow(X)) X$date[wp_idx] else NA
      
      forecasts_4[[length(forecasts_4) + 1L]] <- data.frame(
        target_stock = target,
        date_t       = dates_t[t],
        date_pred    = date_pred_val,
        forecast     = f,
        n_predictors = n_predictors,
        lambda       = lambda_min,
        lambda_max   = lambda_max,
        lambda_norm  = lambda_norm,
        stringsAsFactors = FALSE
      )
      
      nz <- which(b != 0)
      if (length(nz)) {
        coefs_4[[length(coefs_4) + 1L]] <- data.frame(
          target_stock = target,
          date_t       = dates_t[t],
          predictor    = names(b)[nz],
          beta         = as.numeric(b[nz]),
          lambda       = lambda_min,
          lambda_max   = lambda_max,
          lambda_norm  = lambda_norm,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

stopCluster(cl)

cat("Progress: [", rep("=", 50), "] 100.0% - Complete!\n", sep = "")






# --- tidy results ---
forecasts_4 <- if (length(forecasts_4)) do.call(rbind, forecasts_4) else data.frame()
coefs_4     <- if (length(coefs_4))     do.call(rbind, coefs_4)     else data.frame()
results_4 <- list(
  forecasts_4 = as.data.table(forecasts_4),
  active_coefs_4 = as.data.table(coefs_4),
  meta = list(L = L, K_ret = K, K_topic = K, nfolds = nfolds, lambda_choice = "lambda.min")
)