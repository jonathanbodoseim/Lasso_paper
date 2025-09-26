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

load(here("data", "X_data"))

# --- inputs ---
X <- as.data.table(X_data)                
setorder(X, date)
permno_ticker <- colnames(X_data)
topic_cols <- grep("^Topic_", names(X), value = TRUE)
stock_cols <- setdiff(setdiff(names(X), "date"), topic_cols)
target_stocks <- as.character(permno_ticker[2:486])
K <- 3                                                  
L <- 30                                                   
nfolds <- 10                                          
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
  X[, .(date)],
  lag_block(X, stock_cols, K, "R"),
  lag_block(X, topic_cols, K, "T")
)

first_valid <- 1 + (K - 1)                 # first row with all lags available
Xlag <- Xlag[first_valid:.N]
pred_cols <- setdiff(names(Xlag), "date")  # extract names of predictors
Xmat     <- as.matrix(Xlag[, ..pred_cols])
dates_t  <- Xlag$date
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
        alpha = 1, family = "gaussian", nfolds = nfolds,
        standardize = TRUE, parallel = TRUE
      )
      
      # --- lambda bookkeeping (must be defined before using below) ---
      lambda_min  <- fit$lambda.min
      lambda_max  <- fit$lambda[1]
      lambda_norm <- lambda_min / lambda_max
      
      # prediction at lambda.min
      f <- as.numeric(predict(fit, newx = Xmat[t, , drop = FALSE], s = "lambda.min"))
      
      # Count non-zero predictors (excluding intercept)
      b  <- as.matrix(coef(fit, s = "lambda.min"))[-1, , drop = TRUE]
      n_predictors <- sum(b != 0)
      
      # safe date_pred
      wp_idx <- orig_idx[t] + 1L
      date_pred_val <- if (wp_idx <= nrow(X)) X$date[wp_idx] else NA
      
      # save forecast row
      forecasts[[length(forecasts) + 1L]] <- data.frame(
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
      
      # save coefficient rows (only if any active)
      nz <- which(b != 0)
      if (length(nz)) {
        coefs[[length(coefs) + 1L]] <- data.frame(
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

# --- Plot active predictors vs lambda ---

n_points <- nrow(results$forecasts)

lambda_means <- results$forecasts %>%
  filter(n_predictors %in% c(5,10,15,20,25,30,35,40)) %>%
  group_by(n_predictors) %>%
  summarise(mean_lambda = mean(lambda, na.rm = TRUE), .groups = "drop")

# --- Plot ---
n_points <- sum(!is.na(results$forecasts$lambda) & !is.na(results$forecasts$n_predictors))
y_top    <- max(results$forecasts$n_predictors, na.rm = TRUE)
pad      <- diff(range(results$forecasts$n_predictors, na.rm = TRUE)) * 0.04

p <- ggplot(results$forecasts, aes(x = lambda, y = n_predictors)) +
  geom_point(color = "darkred", alpha = 0.6, size = 1.5) +
  # vertical dashed lines
  geom_vline(
    data = lambda_means,
    aes(xintercept = mean_lambda),
    linetype = "dashed", linewidth = 0.5, color = "steelblue"
  ) +
  # labels at the very top of each line
  geom_text(
    data = lambda_means,
    aes(x = mean_lambda, y = y_top + pad, label = n_predictors),
    vjust = 0, size = 3, color = "steelblue", inherit.aes = FALSE
  ) +
  labs(
    title = "Lasso Regularization: Lambda vs Number of Active Predictors",
    subtitle = paste0("Each point = one forecast | N = ", scales::comma(n_points)),
    x = "Lambda",
    y = "Number of Active Predictors"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11, color = "gray50"),
    plot.margin = margin(t = 25)  # extra space on top for labels
  ) +
  scale_x_continuous(trans = "log10", labels = scientific_format()) +
  annotation_logticks(sides = "b")

# --- save results ---
write_parquet(results$forecasts,    file.path(here("output"), "forecasts.parquet"), compression = "zstd")
write_parquet(results$active_coefs, file.path(here("output"), "active_coefs.parquet"), compression = "zstd")
write_json(results$meta,            file.path(here("output"), "meta.json"), pretty = TRUE, auto_unbox = TRUE)

ggsave(
  filename = "output/lambdavspredictors.png",  # file path
  plot = p,                                   # plot object
  width = 8, height = 5, dpi = 300            # dimensions & resolution
)

