# R/utils_lasso.R

#' Run rolling LASSO on returns + topic lags
#'
#' @param X_data data.frame/data.table with a `week` column, stock return columns, and topic columns (prefixed by `topic_pref`).
#' @param target_stocks Character vector of stock column names to forecast.
#' @param week_col Name of the week column. Default "week".
#' @param topic_pref Prefix of topic columns (e.g., "Topic_").
#' @param L Rolling window length (weeks).
#' @param K_ret Number of lags for returns.
#' @param K_topic Number of lags for topics.
#' @param lambda_choice One of "lambda.min" or "lambda.1se".
#' @param nfolds_cv Number of CV folds.
#' @param parallel Use glmnet parallelization (see glmnet docs).
#' @param nonzero_tol Threshold for selecting nonzero coefficients.
#' @param seed Random seed.
#' @param verbose Logical; if TRUE prints progress.
#' @return A list with `forecasts`, `active_coefs`, and `meta`.
#' @export
run_lasso_ret_and_topics <- function(
    X_data,
    target_stocks,                  # character vector of stock columns to forecast
    week_col      = "week",
    topic_pref    = "Topic_",
    L             = 30,             # rolling window length (weeks)
    K_ret         = 3,              # lags for returns: RLag0..RLag2
    K_topic       = 3,              # lags for topics: TLag0..TLag2
    lambda_choice = c("lambda.min","lambda.1se")[1],
    nfolds_cv     = 5,
    parallel      = TRUE,
    nonzero_tol   = 0,
    seed          = 123,
    verbose       = TRUE            # << logging switch
){
  .log <- function(...){ if (verbose) { base::message(sprintf(...)); utils::flush.console() } }
  
  stopifnot(week_col %in% names(X_data))
  t0 <- Sys.time(); .log("▶ Starting run at %s", format(t0))
  data.table::setDT(X_data); data.table::setorderv(X_data, week_col)
  
  topic_cols <- grep(paste0("^", topic_pref), names(X_data), value = TRUE)
  if (!length(topic_cols)) stop("No topic columns found with prefix '", topic_pref, "'.")
  stock_cols <- setdiff(setdiff(names(X_data), c(week_col)), topic_cols)
  if (!all(target_stocks %in% stock_cols)) stop("Some target_stocks not found among stock columns.")
  .log("✓ Identified %d topic cols, %d stock cols; targets: %s", length(topic_cols), length(stock_cols), paste(target_stocks, collapse=", "))
  
  build_lag_block <- function(dt, cols, K, tag){
    lapply(0:(K-1), function(k){
      block <- dt[, data.table::shift(.SD, n = k, type = "lag"), .SDcols = cols]
      data.table::setnames(block, paste0(cols, "_", tag, "Lag", k)); block
    })
  }
  
  .log("… Building lag blocks: returns K=%d, topics K=%d", K_ret, K_topic)
  ret_blocks   <- build_lag_block(X_data, stock_cols, K_ret,   "R")
  topic_blocks <- build_lag_block(X_data, topic_cols, K_topic, "T")
  
  .log("… Assembling feature matrix")
  X_full <- do.call(cbind, c(ret_blocks, topic_blocks))
  X_full <- cbind(X_data[, ..week_col], X_full)
  
  first_valid_row <- 1 + max(K_ret - 1, K_topic - 1)
  X_full <- X_full[first_valid_row:.N] # keep only rows from first valid one to last one
  X_cols <- setdiff(names(X_full), week_col) # build list of predictor columns excluding week
  Xfull_to_orig <- function(i) first_valid_row - 1 + i
  
  t_idx <- L
  lo <- t_idx - L + 1; hi <- t_idx
  X_win <- as.matrix(X_full[lo:hi, X_cols, with = FALSE])
  
  cat("Rows in window:", nrow(X_win), "Cols:", ncol(X_win), "\n")
  
  # Count row completeness BEFORE filtering
  cc <- complete.cases(X_win)
  cat("Complete rows (predictors only):", sum(cc), "of", length(cc), "\n")
  
  # Where are the holes?
  col_na_counts <- colSums(!is.finite(X_win))  # counts NaN/Inf too
  bad_cols <- sort(col_na_counts[col_na_counts > 0], decreasing = TRUE)
  cat("Columns with any non-finite in window:", length(bad_cols), "\n")
  if (length(bad_cols)) print(head(bad_cols, 10))
  
  # Y side
  y_rows_orig <- first_valid_row - 1 + (lo:hi) + 1
  y_win <- X_data[[target_stocks[1]]][y_rows_orig]
  cat("Non-finite in y_win:", sum(!is.finite(y_win)), "\n")
  
  # Final keep
  keep <- complete.cases(X_win) & is.finite(y_win)
  cat("Rows kept for fit:", sum(keep), "\n")
  
  all_forecasts <- list(); all_active_coefs <- list()
  set.seed(seed)
  
  s <- match.arg(lambda_choice, c("lambda.min","lambda.1se"))
  .log("… Rolling LASSO (L=%d, CV folds=%d, lambda=%s, parallel=%s)", L, nfolds_cv, s, parallel)
  
  for (target in target_stocks) {  # loop over all stocks
    .log("→ Target: %s", target)
    target_res <- list(); target_coef <- list()
    max_t_idx <- nrow(X_full) - 1
    if (max_t_idx < L) { .log("  (skipped: not enough rows for window)"); next }
    
    pb <- utils::txtProgressBar(min=L, max=max_t_idx, style=3)
    for (t_idx in L:max_t_idx) {
      if (verbose) utils::setTxtProgressBar(pb, t_idx)
      
      lo <- t_idx - L + 1; hi <- t_idx
      X_win <- as.matrix(X_full[lo:hi, X_cols, with = FALSE])
      
      y_rows_orig <- Xfull_to_orig(lo:hi) + 1
      y_win <- X_data[[target]][y_rows_orig]
      
      # Sanity check: y_win should equal a lead(1) of target aligned to X_full
      y_check <- X_data[[target]][Xfull_to_orig(lo:hi) + 1]
      if (!all.equal(y_win, y_check, check.attributes = FALSE)) {
        warning(sprintf("Lead-1 misalignment detected at target=%s, window ending t_idx=%d", target, t_idx))
      }
      
      keep <- stats::complete.cases(X_win) & !is.na(y_win)
      X_win_k <- X_win[keep, , drop = FALSE]; y_win_k <- y_win[keep]
      if (nrow(X_win_k) < 5) next
      
      cvfit <- glmnet::cv.glmnet(x = X_win_k, y = y_win_k, family = "gaussian", alpha = 1,
                                 nfolds = nfolds_cv, parallel = parallel, standardize = TRUE)
      
      coefs <- as.matrix(stats::coef(cvfit, s = s))
      intercept <- as.numeric(coefs[1, , drop = TRUE])
      betas <- coefs[-1, , drop = TRUE]; rn <- rownames(coefs)[-1]
      active_idx <- which(abs(betas) > nonzero_tol)
      
      x_t  <- as.matrix(X_full[t_idx, X_cols, with = FALSE])
      f_t1 <- as.numeric(stats::predict(cvfit, newx = x_t, s = s))
      
      week_t    <- X_full[[week_col]][t_idx]
      week_pred <- X_data[[week_col]][ Xfull_to_orig(t_idx) + 1 ]
      
      target_res[[length(target_res) + 1]] <- data.table::data.table(
        target_stock = target, week_t = week_t, week_pred = week_pred,
        forecast = f_t1, intercept = intercept,
        lambda = if (s=="lambda.min") cvfit$lambda.min else cvfit$lambda.1se,
        n_active = length(active_idx)
      )
      
      if (length(active_idx) > 0) {
        target_coef[[length(target_coef) + 1]] <- data.table::data.table(
          target_stock = target, week_t = week_t,
          predictor = rn[active_idx], beta = as.numeric(betas[active_idx])
        )
      }
    }
    if (verbose) close(pb)
    
    all_forecasts[[target]]    <- if (length(target_res))  data.table::rbindlist(target_res)  else data.table::data.table()
    all_active_coefs[[target]] <- if (length(target_coef)) data.table::rbindlist(target_coef) else data.table::data.table()
    .log("✓ Done target %s: %d forecasts, %d non-zero predictors", target,
         nrow(all_forecasts[[target]]), nrow(all_active_coefs[[target]]))
  }
  
  t1 <- Sys.time(); .log("✔ Finished in %.2f sec", as.numeric(difftime(t1, t0, units="secs")))
  
  list(
    forecasts    = data.table::rbindlist(all_forecasts, fill = TRUE),
    active_coefs = data.table::rbindlist(all_active_coefs, fill = TRUE),
    meta = list(L=L, K_ret=K_ret, K_topic=K_topic, lambda_choice=s,
                predictors="3 lags of ALL returns + 3 lags of ALL topics",
                started=t0, finished=t1)
  )
}


