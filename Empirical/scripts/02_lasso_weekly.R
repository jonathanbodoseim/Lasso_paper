suppressPackageStartupMessages({
  library(data.table)
  library(glmnet)
  library(doParallel)
  library(readr)  # for read_csv()
  library(here)   # project-rooted paths
  library(ggplot2)
})

X_data <- read_csv(here("output", "X_data.csv"), show_col_types = FALSE)
permno_ticker <- read_csv(here("output", "permno.csv"), show_col_types = FALSE)

# --- inputs ---
X <- as.data.table(X_data)                # your table shown above
setorder(X, week)
topic_cols <- grep("^Topic_", names(X), value = TRUE)
stock_cols <- setdiff(setdiff(names(X), "week"), topic_cols)
target_stocks <- as.character(permno_ticker$PERMNO[1:2])  # <- define your targets here
K <- 3                                                    # 3 lags each
L <- 30                                                   # rolling window length
nfolds <- 10                                              # CV folds
set.seed(1)

# --- build 3 lags for returns and topics ---
lag_block <- function(dt, cols, K, tag) {
  if (!length(cols)) return(NULL)
  do.call(cbind, lapply(0:(K-1), function(k) {
    out <- dt[, shift(.SD, n = k), .SDcols = cols]
    setnames(out, paste0(cols, "_", tag, "Lag", k))
    out
  }))
}

Xlag <- cbind(
  X[, .(week)],
  lag_block(X, stock_cols, K, "R"),
  lag_block(X, topic_cols, K, "T")
)

first_valid <- 1 + (K - 1)                 # first row with all lags available
Xlag <- Xlag[first_valid:.N]
pred_cols <- setdiff(names(Xlag), "week")  # extract names of predictors
Xmat     <- as.matrix(Xlag[, ..pred_cols])
weeks_t  <- Xlag$week
orig_idx <- first_valid:nrow(X)            # map back to original X rows

# --- rolling lasso per target ---
# setup parallelization
n_cores <- parallel::detectCores(logical = TRUE)
cl <- makeCluster(max(1L, n_cores - 1L))   # leave 1 core free
registerDoParallel(cl)

forecasts <- list()
coefs     <- list()

# setup progress bar
total_iterations <- 0 
for (target in target_stocks) {
  max_t <- nrow(Xmat) - 1L
  if (max_t >= L) {
    total_iterations <- total_iterations + (max_t - L)
  }
}

# Initialize progress tracking
current_iteration <- 0
cat("Running rolling lasso for", length(target_stocks), "stocks...\n")
cat("Progress: [", rep(" ", 50), "]\r", sep = "")

for (target in target_stocks) {
  max_t <- nrow(Xmat) - 1L
  if (max_t < L) next
  
  for (t in (L+1):max_t) {
    
    # Update progress bar
    current_iteration <- current_iteration + 1
    progress_pct <- current_iteration / total_iterations
    filled_bars <- round(progress_pct * 50)
    cat("Progress: [", rep("=", filled_bars), rep(" ", 50 - filled_bars), "] ", 
        round(progress_pct * 100, 1), "%\r", sep = "")
    flush.console()
    
    lo <- t - L; hi <- t - 1          # excludes Xmat[t,] from training  
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
        alpha = 1, family = "gaussian", nfolds = nfolds, standardize = TRUE, parallel = TRUE
      )
      
      f <- as.numeric(predict(fit, newx = Xmat[t, , drop = FALSE], s = "lambda.min"))
      
      # Count non-zero predictors (excluding intercept)
      b  <- as.matrix(coef(fit, s = "lambda.min"))[-1, , drop = TRUE]
      n_predictors <- sum(b != 0)
      
      forecasts[[length(forecasts) + 1L]] <- data.frame(
        target_stock = target,
        week_t       = weeks_t[t],
        week_pred    = X$week[orig_idx[t] + 1L],  # (kept as in your original)
        forecast     = f,
        n_predictors = n_predictors,
        lambda       = fit$lambda.min,
        stringsAsFactors = FALSE
      )
      
      nz <- which(b != 0)
      if (length(nz)) {
        coefs[[length(coefs) + 1L]] <- data.frame(
          target_stock = target,
          week_t       = weeks_t[t],
          predictor    = names(b)[nz],
          beta         = as.numeric(b[nz]),
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

stopCluster(cl)

# Complete progress bar
cat("Progress: [", rep("=", 50), "] 100.0% - Complete!\n", sep = "")

# --- tidy results ---
forecasts <- if (length(forecasts)) do.call(rbind, forecasts) else data.frame()
coefs     <- if (length(coefs))     do.call(rbind, coefs)     else data.frame()

results <- list(
  forecasts = as.data.table(forecasts),
  active_coefs = as.data.table(coefs),
  meta = list(L = L, K_ret = K, K_topic = K, nfolds = nfolds, lambda_choice = "lambda.min")
)



